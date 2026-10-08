#!/usr/bin/env python3
"""Scene-chunked, VMAF-targeted AV1 encoding with QSVEncC (Intel Arc).

For each input file:
  1. Detect scene changes with ffmpeg (cached under <work>/scene_caches).
  2. Turn them into cut points at least --min-len and at most --max-len
     seconds apart, and split the first video track into chunks (stream copy).
  3. Encode each chunk with QSVEncC in ICQ mode, searching for the highest ICQ
     (smallest file) whose VMAF still meets --target.
  4. Concatenate the chunks and mux them with the source's audio (re-encoded
     to Opus), subtitles, attachments, chapters and metadata.
  5. Check that the output has as many video frames as the source.

Finished chunks are kept in the work directory, so re-running the same command
after an interruption resumes where it left off.
"""
import argparse, csv, glob, json, math, os, re, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Approximate VMAF change per ICQ step (measured ~0.65-0.9). Used to estimate
# how far to move the ICQ before the search has both a pass and a fail.
SLOPE = 0.8
VMAF_RE = re.compile(r"VMAF Score\s+([0-9]+(?:\.[0-9]+)?)")  # QSVEncC log line
PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")       # ffmpeg metadata=print line
SW_CODECS = {"vc1", "wmv3"}                      # QSV can't hw-decode these
OPUS_BR = {1: 64, 2: 128, 6: 256, 8: 384}        # Opus kbps by channel count; 64/channel otherwise
# libopus accepts exactly one channel layout per channel count. Other layouts
# (e.g. "5.1(side)" from AC3/DTS, "7.1(wide)") are remapped to these.
OPUS_LAYOUTS = {1: "mono", 2: "stereo", 3: "3.0", 4: "quad",
                5: "5.0", 6: "5.1", 7: "6.1", 8: "7.1"}
A = None                  # parsed command-line arguments, set in main()
PLOCK = threading.Lock()  # serializes console output from worker threads


def log(msg):
    """Print a line from any thread, blanking any in-place progress bar first."""
    with PLOCK:
        sys.stdout.write("\r" + " " * 85 + f"\r{msg}\n")
        sys.stdout.flush()


def run(cmd, cwd=None):
    """Run a command to completion, capturing stdout/stderr as text."""
    return subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def probe(path):
    """Return ffprobe's stream and format info for a file as a dict."""
    r = run([A.ffprobe, "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", path])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.strip()}")
    return json.loads(r.stdout)


# ---------------------------------------------------------------- scenes/split
def detect_scenes(src, cache):
    """Return the timestamps (seconds) of scene changes in src.

    This decodes the whole video, so the result is cached as JSON in `cache`
    and reused on later runs.
    """
    if cache.exists():
        return json.loads(cache.read_text())

    # Estimate the frame count. Only used to scale the progress bar.
    probe_info = probe(src)
    vstream = next((s for s in probe_info["streams"] if s["codec_type"] == "video"), {})
    total_frames = int(vstream.get("nb_frames") or 0)
    if total_frames <= 0:
        try:
            duration = float(probe_info["format"]["duration"])
            fr_parts = vstream.get("avg_frame_rate", "24/1").split("/")
            fps = float(fr_parts[0]) / float(fr_parts[1]) if len(fr_parts) > 1 else float(fr_parts[0])
            total_frames = int(duration * fps)
        except Exception:
            total_frames = 179040  # arbitrary fallback (~2 h at 24 fps)

    # showinfo comes before select so it logs every frame to stderr (progress).
    # select keeps frames whose scene score exceeds the threshold, and
    # metadata=print writes their pts_time to stdout.
    vf = f"scale=640:-2,showinfo,select='gt(scene,{A.scene_thresh})',metadata=print:file=-"
    cmd = [A.ffmpeg, "-hide_banner", "-loglevel", "info",
           "-an", "-sn", "-dn",
           "-i", src, "-map", "0:v:0", "-vf", vf, "-f", "null", "-"]
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    # Drain stdout on its own thread so neither pipe can fill up and stall ffmpeg.
    raw_output = []
    def read_stdout(pipe, lst):
        for line in pipe:
            lst.append(line)
    stdout_thread = threading.Thread(target=read_stdout, args=(proc.stdout, raw_output))
    stdout_thread.start()

    bar_len = 30
    while True:
        line = proc.stderr.readline()
        if not line and proc.poll() is not None:
            break
        if not line:
            continue
        # e.g. "[Parsed_showinfo_1 @ 0x...] n:  42 pts: 43043 pts_time:1.793 ..."
        if "showinfo" in line and "n:" in line:
            try:
                parts = line.split("n:")
                if len(parts) > 1:
                    curr_frame = int(parts[1].strip().split()[0])
                    pct = min(100.0, max(0.0, (curr_frame / total_frames) * 100))
                    filled = min(bar_len, max(0, int(round(bar_len * curr_frame / total_frames))))
                    bar = '█' * filled + '-' * (bar_len - filled)
                    sys.stdout.write(f"\rSCENE DETECT: |{bar}| {pct:.1f}% (Frame {curr_frame}/{total_frames})   ")
                    sys.stdout.flush()
            except (ValueError, IndexError):
                pass

    stdout_thread.join()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"scene detection failed: process exited with code {proc.returncode}")
    print("\r" + " " * 80 + "\rSCENE DETECT: Finished analysis.")

    times = [float(x) for x in PTS_RE.findall("".join(raw_output))]
    cache.write_text(json.dumps(times))
    return times


def forced_splits(a, b, max_len):
    """Evenly spaced cut points splitting [a, b] into pieces of at most max_len."""
    n = math.ceil((b - a) / max_len)
    return [a + (b - a) * k / n for k in range(1, n)] if n > 1 else []


def plan_cuts(times, duration):
    """Choose chunk boundaries (seconds) from scene-change times.

    Scene changes less than --min-len after the previous cut, or before the end
    of the file, are skipped. Spans longer than --max-len get extra, evenly
    spaced cuts.
    """
    cuts, last = [], 0.0
    for t in sorted(times):
        if t - last >= A.min_len and duration - t >= A.min_len:
            cuts += forced_splits(last, t, A.max_len)
            cuts.append(t)
            last = t
    cuts += forced_splits(last, duration, A.max_len)
    return cuts


def split_source(src, segdir, cuts):
    """Split the first video track of src into segdir/chunk_NNNNN.mkv.

    This is a stream copy, so each chunk really starts at the first keyframe at
    or after its requested cut. The actual boundaries are read back from the
    segment list ffmpeg writes. chunks.csv is only renamed into place once the
    split succeeds, so its presence means the split can be skipped.

    Returns [{"name", "start", "end"}, ...] in playback order.
    """
    final_csv = segdir / "chunks.csv"
    if not final_csv.exists():
        probe_info = probe(src)
        try:
            total_duration = float(probe_info["format"]["duration"])
        except (KeyError, ValueError, TypeError):
            total_duration = 1.0  # only used to scale the progress bar

        # -progress pipe:1 makes ffmpeg write key=value progress lines to stdout.
        cmd = [A.ffmpeg, "-hide_banner", "-loglevel", "error", "-progress", "pipe:1", "-y", "-i", src.resolve(),
               "-map", "0:v:0", "-c", "copy", "-an", "-sn", "-dn",
               "-f", "segment", "-segment_format", "matroska", "-reset_timestamps", "1",
               "-segment_list", "chunks.tmp.csv", "-segment_list_type", "csv"]
        cmd += (["-segment_times", ",".join(f"{t:.3f}" for t in cuts)] if cuts
                else ["-segment_time", "1000000"])
        cmd += ["chunk_%05d.mkv"]
        proc = subprocess.Popen([str(c) for c in cmd], cwd=segdir, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

        bar_len = 30
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                continue
            # Current output position, in microseconds.
            if line.startswith("out_time_us="):
                try:
                    curr_time = int(line.split("=")[1].strip()) / 1000000.0
                    pct = min(100.0, max(0.0, (curr_time / total_duration) * 100))
                    filled = min(bar_len, max(0, int(round(bar_len * curr_time / total_duration))))
                    bar = '█' * filled + '-' * (bar_len - filled)
                    sys.stdout.write(f"\rSPLITTING:    |{bar}| {pct:.1f}% ({curr_time:.1f}/{total_duration:.1f}s)   ")
                    sys.stdout.flush()
                except (ValueError, IndexError):
                    pass

        _, stderr_val = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"split failed: {stderr_val.strip()}")
        print("\r" + " " * 80 + "\rSPLITTING:    Finished splitting source video.")
        os.replace(segdir / "chunks.tmp.csv", final_csv)

    rows = []
    with open(final_csv, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                rows.append({"name": row[0], "start": float(row[1]), "end": float(row[2])})
    return rows


# ------------------------------------------------------------------- encoding
class Ctx:
    """Per-file state shared by the chunk worker threads.

    sw:   decode the remaining chunks in software. Always starts False here (the
          constructor argument is ignored), and becomes True once a software
          retry succeeds. Note that run_qsv doesn't act on it yet.
    hint: ICQ to start the next chunk's search from. It's a running average of
          the ICQs chosen so far, so later chunks need fewer trial encodes.
    """
    def __init__(self, sw):
        self.sw = False
        self.hint = A.qstart


def run_qsv(chunk, q, out, lg, sw, vthreads):
    """Encode one chunk at ICQ q and measure its VMAF with QSVEncC's --vmaf.

    Returns (vmaf, output_bytes), or None on failure, in which case the command
    and its output are appended to error_debug.log next to the log file.
    `sw` is meant to select software decoding, but the input is currently
    always decoded in hardware (--avhw).
    """
    cmd = [A.qsvencc, "--device", "1", "--avhw", "-i", chunk,
           "-c", "av1", "--icq", q, "--gop-len", A.gop, "--quality", "1",
           "--output-depth", 10, "--colorrange", "auto", "--colormatrix",
           "auto", "--colorprim", "auto", "--transfer", "auto",
           "--chromaloc", "auto", "--tune", "perceptual", "--pic-struct",
           "--avsync", "vfr"]

    # Per-era cleanup filters.
    if A.era == "1":
        cmd += ["--vpp-nlmeans", "d=1,search_t=7"]
    elif A.era == "2":
        cmd += ["--vpp-nlmeans", "d=1,search_t=7"]
    elif A.era == "3":
        cmd += ["--vpp-hqdn3d", "luma_spatial=2.5,chroma_spatial=2.0,luma_temporal=4.0,chroma_temporal=3.0",
                "--vpp-unsharp", "radius=3,weight=0.5"]

    # The VMAF score is read back from the log file.
    cmd += ["--vmaf", f"subsample={A.subsample},threads={vthreads}",
            "--log", lg, "-o", out]

    r = run(cmd)
    text = lg.read_text(encoding="utf-8", errors="replace") if lg.exists() else ""
    m = VMAF_RE.search(text)
    if r.returncode != 0 or not m or not out.exists():
        with open(lg.parent / "error_debug.log", "a", encoding="utf-8") as dbg:
            dbg.write(f"\n--- CHUNK FAILED (RC={r.returncode}) ---\nCMD: {' '.join(str(x) for x in cmd)}\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}\n")
        return None
    return float(m.group(1)), out.stat().st_size


def try_encode(ctx, chunk, q, trydir, idx, vthreads):
    """Encode chunk idx at ICQ q, retrying with software decode if the first try fails.

    Era 1, or a ctx already switched to software decode, makes a single
    software attempt. Returns (vmaf, bytes, path), or raises RuntimeError with
    the tail of error_debug.log if every attempt fails.
    """
    out = trydir / f"chunk_{idx:05d}_q{q}.mkv"
    lg = trydir / f"chunk_{idx:05d}_q{q}.log"

    initial_sw = True if ctx.sw else False

    for sw in ([True] if initial_sw else [False, True]):
        res = run_qsv(chunk, q, out, lg, sw, vthreads)
        if res:
            if sw:
                ctx.sw = True
            return res[0], res[1], out
    tail = (trydir / "error_debug.log").read_text(errors="replace")[-1000:] if (trydir / "error_debug.log").exists() else "No diagnostic console trace found."
    raise RuntimeError(f"chunk {idx} q={q} failed:\n{tail}")


def search_chunk(ctx, idx, chunk, trydir, encdir, vthreads):
    """Find the highest ICQ (smallest file) whose VMAF meets --target for one chunk.

    Starts at ctx.hint. Until there's both a passing and a failing ICQ, it steps
    by the VMAF surplus/shortfall divided by SLOPE. After that, it interpolates
    linearly between the closest pass and fail. It stops when they're adjacent,
    when it hits --qmin/--qmax, or after --max-iters rounds. If nothing meets
    the target, the chunk is encoded at --qmin.

    The chosen encode is moved to encdir with a chunk_NNNNN.json record (used
    for resuming and the report). The other trial encodes are deleted.
    """
    start_time = time.perf_counter()
    tried = {}  # icq -> (vmaf, bytes, path)

    def get(q):
        if q not in tried:
            tried[q] = try_encode(ctx, chunk, q, trydir, idx, vthreads)
        return tried[q]

    target, qmin, qmax = A.target, A.qmin, A.qmax
    q = max(qmin, min(qmax, ctx.hint))
    best_pass = best_fail = None  # highest passing / lowest failing ICQ so far
    for _ in range(A.max_iters):
        score = get(q)[0]
        if score >= target:
            best_pass = q if best_pass is None else max(best_pass, q)
        else:
            best_fail = q if best_fail is None else min(best_fail, q)
        if best_pass is not None and best_fail is not None:
            gap = best_fail - best_pass
            if gap <= 1:
                break
            sp, sf = tried[best_pass][0], tried[best_fail][0]
            frac = (sp - target) / (sp - sf) if sp > sf else 0.5
            q = best_pass + max(1, min(gap - 1, round(frac * gap)))
        elif best_pass is not None:
            if best_pass >= qmax:
                break
            q = min(qmax, best_pass + max(1, round((score - target) / SLOPE)))
        else:
            if best_fail <= qmin:
                break
            q = max(qmin, best_fail - max(1, round((target - score) / SLOPE)))

    if best_pass is not None:
        chosen = best_pass
    else:
        chosen = qmin
        get(qmin)
    score, size, path = tried[chosen]
    os.replace(path, encdir / f"chunk_{idx:05d}.mkv")
    for qq, (_, _, p) in tried.items():
        if qq != chosen:
            p.unlink(missing_ok=True)

    meta = {"idx": idx, "q": chosen, "vmaf": score, "bytes": size,
            "encodes": len(tried), "met_target": best_pass is not None,
            "duration": time.perf_counter() - start_time}
    (encdir / f"chunk_{idx:05d}.json").write_text(json.dumps(meta))
    ctx.hint = round((ctx.hint + chosen) / 2)
    return meta


# ------------------------------------------------------------------ mux/verify
def count_packets(path):
    """Number of packets (frames) in the first video stream, or -1 if unknown."""
    r = run([A.ffprobe, "-v", "error", "-count_packets", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path])
    m = re.search(r"\d+", r.stdout)
    return int(m.group()) if m else -1


def opus_args(i, stream):
    """ffmpeg output options that encode output audio stream i (from `stream`) to Opus."""
    ch = int(stream.get("channels") or 2)
    args = [f"-c:a:{i}", "libopus", f"-b:a:{i}", f"{OPUS_BR.get(ch, 64 * ch)}k"]
    # async=1 fills gaps in the source audio's timestamps with silence and trims
    # overlaps (anything over 0.1 s), so glitches can't shift later audio out of sync.
    filters = ["aresample=async=1"]
    layout, target = stream.get("channel_layout"), OPUS_LAYOUTS.get(ch)
    if target and layout and layout not in (target, "unknown"):
        filters.append(f"aformat=channel_layouts={target}")
    return args + [f"-filter:a:{i}", ",".join(filters)]


def mux_final(src, video, out, info, vstart):
    """Combine the concatenated AV1 video with everything else from the source.

    Video is stream-copied, audio is re-encoded to Opus, and subtitles,
    attachments (e.g. fonts), chapters and global metadata are copied. Stream
    tags (language, title, ...) and disposition flags (default, forced, ...)
    are carried over from the matching source streams.

    The source is opened twice: input 1 supplies only the audio, and input 2
    supplies subtitles, attachments, chapters and metadata. ffmpeg always reads
    the input that feeds whichever output stream is furthest behind, and between
    subtitle events that's the subtitle stream. If subtitles shared an input with
    the audio, ffmpeg would keep encoding audio while the video input sat idle,
    and the audio would be written several seconds ahead of its video. Players
    seek by jumping to a video keyframe and then find no audio near it, so audio
    drops out after seeking.
    """
    streams = info.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    n_attach = sum(1 for s in streams if s.get("codec_type") == "attachment")

    # The chunks were cut with reset timestamps, so the concatenated video starts
    # at 0. ffmpeg also shifts the source so its earliest stream starts at 0, so
    # offset the video to keep its original position relative to the audio.
    offset = vstart - float(info.get("format", {}).get("start_time") or 0)

    cmd = [A.ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if abs(offset) > 1e-6:
        cmd += ["-itsoffset", f"{offset:.6f}"]
    cmd += ["-i", video, "-i", src, "-i", src,
            "-map", "0:v:0", "-map", "1:a?", "-map", "2:s?", "-map", "2:t?",
            "-map_chapters", "2", "-map_metadata", "2",
            "-map_metadata:s:v:0", "2:s:v:0"]

    # Any per-stream -map_metadata turns off ffmpeg's automatic per-stream tag
    # copying, so every stream's tags are mapped explicitly. Attachments need
    # theirs, because the MKV muxer rejects attachments without a filename tag.
    for i in range(len(audio)):
        cmd += [f"-map_metadata:s:a:{i}", f"1:s:a:{i}"]
    for i in range(len(subs)):
        cmd += [f"-map_metadata:s:s:{i}", f"2:s:s:{i}"]
    for i in range(n_attach):
        cmd += [f"-map_metadata:s:t:{i}", f"2:s:t:{i}"]

    cmd += ["-c:v", "copy", "-c:s", "copy"]
    for i, s in enumerate(audio):
        cmd += opus_args(i, s)

    # Copy each stream's full set of flags ("0" if none). Setting every stream
    # explicitly also stops ffmpeg from choosing a default track on its own.
    for kind, group in (("a", audio), ("s", subs)):
        for i, s in enumerate(group):
            flags = [k for k, v in s.get("disposition", {}).items() if v == 1]
            cmd += [f"-disposition:{kind}:{i}", "+".join(flags) or "0"]

    cmd += [out]
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"final mux failed: {r.stderr.strip()}")


# ---------------------------------------------------------------------- driver
def process(src):
    """Run the whole pipeline for one source file.

    Skips files with no video, files that are already AV1 (unless --force), and
    files whose output already exists.
    """
    src = Path(src)
    info = probe(src)
    vs = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        log(f"SKIP {src.name}: no video stream")
        return
    codec = vs["codec_name"]
    if codec == "av1" and not A.force:
        log(f"SKIP {src.name}: already AV1 (use --force to re-encode)")
        return
    out = Path(A.outdir) / (src.stem + ".mkv")
    if out.exists():
        log(f"SKIP {src.name}: output exists")
        return
    t0 = time.time()
    duration = float(info["format"]["duration"])
    work = Path(A.work) / src.stem
    segdir, trydir, encdir = work / "src", work / "try", work / "enc"
    for d in (segdir, trydir, encdir, Path(A.outdir)):
        d.mkdir(parents=True, exist_ok=True)
    log(f"\n=== {src.name} ({codec}, {duration/60:.1f} min) ===")
    log(f"Active Profile Preset: {A.era}")

    # The scene cache lives outside the per-file work dir, so it survives
    # the work dir being deleted after a successful encode.
    scene_cache_dir = Path(A.work) / "scene_caches"
    scene_cache_dir.mkdir(parents=True, exist_ok=True)
    scene_cache_file = scene_cache_dir / f"{src.stem}_scenes.json"

    log("Detecting scenes...")
    times = detect_scenes(src, scene_cache_file)
    cuts = plan_cuts(times, duration)
    log("Splitting (stream copy)...")
    chunks = split_source(src, segdir, cuts)
    lens = [c["end"] - c["start"] for c in chunks]
    log(f"{len(chunks)} chunks, length min/avg/max = "
        f"{min(lens):.1f}/{sum(lens)/len(lens):.1f}/{max(lens):.1f} s")
    if A.plan_only:
        return

    ctx = Ctx(sw=codec in SW_CODECS)
    if ctx.sw:
        log(f"Enforcing software decode path for stabilization.")
    vthreads = max(1, (os.cpu_count() or 4) // A.workers)  # VMAF threads per worker

    # Reuse chunks that already have both an encode and a .json record.
    results, pending = {}, []
    for i in range(len(chunks)):
        mp = encdir / f"chunk_{i:05d}.json"
        if mp.exists() and (encdir / f"chunk_{i:05d}.mkv").exists():
            results[i] = json.loads(mp.read_text())
        else:
            pending.append(i)
    if results:
        log(f"Resuming: {len(results)} chunks already done.")
    done = len(results)

    with ThreadPoolExecutor(max_workers=A.workers) as ex:
        futs = [ex.submit(search_chunk, ctx, i, segdir / chunks[i]["name"],
                          trydir, encdir, vthreads) for i in pending]
        for fut in as_completed(futs):
            m = fut.result()
            results[m["idx"]] = m
            done += 1
            flag = "" if m["met_target"] else "  (target NOT met at qmin)"
            took_str = f"{int(round(m['duration']))}s" if "duration" in m else "N/A"
            log(f"[{done:03d}/{len(chunks):03d}] chunk {m['idx']:03d}: ICQ {m['q']}, "
                f"VMAF {m['vmaf']:.2f}, {m['bytes']/1e6:.1f} MB, "
                f"{m['encodes']} encodes, took {took_str}{flag}")

            # Overall progress bar, redrawn in place under the per-chunk lines.
            bar_len = 30
            filled = int(round(bar_len * done / len(chunks)))
            percents = round(100.0 * done / len(chunks), 1)
            bar = '█' * filled + '-' * (bar_len - filled)
            sys.stdout.write(f"\rPROGRESS: |{bar}| {percents}% Complete ({done}/{len(chunks)} Chunks)")
            sys.stdout.flush()

    print("\nEncoding pass finished. Moving to combination stages...")

    # Per-chunk report and duration-weighted summary.
    rep = work / "report.csv"
    with open(rep, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk", "start", "end", "icq", "vmaf", "bytes"])
        for i, c in enumerate(chunks):
            m = results[i]
            w.writerow([i, c["start"], c["end"], m["q"], f"{m['vmaf']:.3f}", m["bytes"]])
    wsum = sum(lens)
    avg_vmaf = sum(results[i]["vmaf"] * lens[i] for i in range(len(chunks))) / wsum
    avg_q = sum(results[i]["q"] * lens[i] for i in range(len(chunks))) / wsum
    tot = sum(results[i]["bytes"] for i in range(len(chunks)))
    log(f"Chunk summary: duration-weighted ICQ {avg_q:.1f}, VMAF {avg_vmaf:.2f}, "
        f"min VMAF {min(r['vmaf'] for r in results.values()):.2f}, video {tot/1e6:.0f} MB")

    # Join the chunks with ffmpeg's concat demuxer (stream copy). Single quotes
    # in paths are escaped for its file '...' syntax.
    log("Concatenating...")
    lst = work / "concat.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for i in range(len(chunks)):
            p = (encdir / f"chunk_{i:05d}.mkv").resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{p}'\n")
    video = work / "video_av1.mkv"
    r = run([A.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", lst, "-c", "copy", video])
    if r.returncode != 0:
        raise RuntimeError(f"concat failed: {r.stderr.strip()}")

    log("Muxing audio/subs/chapters...")
    vstart = float(vs.get("start_time") or 0)
    mux_final(src, video, out, info, vstart)

    # A frame-count mismatch means frames were dropped or duplicated somewhere
    # (e.g. at chunk boundaries or by --avsync forcecfr), which can cause A/V drift.
    a, b = count_packets(src), count_packets(out)
    if a != b:
        log(f"WARNING: video packet count differs (source {a}, output {b}). "
            f"Check {out.name} for missing/extra frames or sync problems.")
    else:
        log(f"Verified: {a} video frames in source and output.")

    log(f"Output: {out} ({out.stat().st_size/1e6:.0f} MB; source {src.stat().st_size/1e6:.0f} MB) "
        f"in {(time.time()-t0)/60:.1f} min")
    if not A.keep_work:
        shutil.rmtree(work, ignore_errors=True)


def main():
    global A
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="files, directories (all *.mkv inside) or wildcards")
    p.add_argument("--era", choices=["1", "2", "3"], required=True,
                   help="filter preset: 1 = keep source timestamps (VFR) + NL-means denoise, "
                        "2 = CFR + NL-means denoise, 3 = CFR + hqdn3d denoise + unsharp")
    p.add_argument("--outdir", default="out_av1", help="where finished files are written")
    p.add_argument("--work", default="work",
                   help="temporary files (per-file subfolders) and the scene-detection cache")
    p.add_argument("--target", type=float, default=94.0, help="per-chunk VMAF target")
    p.add_argument("--qmin", type=int, default=16,
                   help="lowest (highest quality) ICQ the search may use")
    p.add_argument("--qmax", type=int, default=34,
                   help="highest (lowest quality) ICQ the search may use")
    p.add_argument("--qstart", type=int, default=24, help="ICQ the first chunk's search starts at")
    p.add_argument("--max-iters", type=int, default=6, help="max search rounds per chunk")
    p.add_argument("--workers", type=int, default=2, help="chunks encoded in parallel")
    p.add_argument("--subsample", type=int, default=3,
                   help="score every Nth frame for VMAF (higher is faster, less precise)")
    p.add_argument("--gop", type=int, default=480, help="max keyframe interval in frames")
    p.add_argument("--scene-thresh", type=float, default=0.30,
                   help="ffmpeg scene-change score (0-1) that counts as a cut")
    p.add_argument("--min-len", type=float, default=6.0, help="min chunk length in seconds")
    p.add_argument("--max-len", type=float, default=60.0, help="max chunk length in seconds")
    p.add_argument("--plan-only", action="store_true",
                   help="detect scenes and split, then stop before encoding")
    p.add_argument("--keep-work", action="store_true",
                   help="keep the work folder after a successful encode")
    p.add_argument("--force", action="store_true", help="also re-encode AV1 sources")
    p.add_argument("--qsvencc", default="QSVEncC64", help="QSVEncC executable")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    p.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    A = p.parse_args()

    files = []
    for pat in A.inputs:
        if os.path.isdir(pat):
            files += sorted(glob.glob(os.path.join(pat, "*.mkv")))
        else:
            files += sorted(glob.glob(pat)) or [pat]
    if not files:
        sys.exit("No input files found.")
    try:
        for f in files:
            try:
                process(f)
            except Exception as e:
                log(f"ERROR on {f}: {e}")
    except KeyboardInterrupt:
        log("\nInterrupted. Re-run the same command to resume finished chunks.")


if __name__ == "__main__":
    main()
