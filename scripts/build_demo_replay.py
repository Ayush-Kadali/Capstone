"""Create a silent video replay from actual CLI output. Optional: Pillow and ffmpeg.

Run scripts/run_demo.sh first. This is a presentation of captured output, not a
live screen recording. The original output files remain alongside the video.
"""
from pathlib import Path
import subprocess
import textwrap
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
source = root / "artifacts/segfault"
frames = source / "replay-frames"
frames.mkdir(exist_ok=True)
font_candidates = [
    Path("/System/Library/Fonts/Menlo.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
]
font_path = next((p for p in font_candidates if p.exists()), None)
if font_path is None:
    raise SystemExit("Install a monospace font or set font_path in this optional presentation script.")

def font(size):
    return ImageFont.truetype(str(font_path), size)

def table(filename):
    text = (source / filename).read_text()
    return text[text.index("POLICY "):text.index("\nBounded modeled optimum:")].strip()

fragmented = (source / "fragmented.txt").read_text()
commands = fragmented[fragmented.index("Command plan ACC"):fragmented.index("\nOutput (row-major)")].strip()
scenes = [
    ("PlacementLens", "An explainable compiler pipeline for AI graph placement", "ACTUAL CLI OUTPUT REPLAY\n\nTyped graph -> capability analysis -> placement\n            -> transfer insertion -> lowering\n            -> checked software execution\n\nMatMul -> ReLU -> Add\n\nC = CPU     A = functional virtual accelerator", "The virtual device runs real arithmetic in software. Costs are illustrative and uncalibrated.", 12),
    ("01 / Full device support", "All three operators can form one device region", "$ placementlens compare --graph examples/fragmented.plg\n  --relu-supported\n\n" + table("full-support.txt"), "AAA is the modeled optimum. Every selected plan passes the independent reference check.", 18),
    ("02 / Remove ReLU support", "More offloading can add costly boundaries", "$ placementlens compare --graph examples/fragmented.plg\n\n" + table("fragmented.txt") + "\n\nACA: 112 transferred bytes, two device runs\nACC:  64 transferred bytes, one device run", "The cost-based plan keeps Add on the CPU. These numbers describe the declared model, not hardware speedups.", 22),
    ("03 / Inspect the lowered commands", "These same commands are exported and executed", commands, "H: and D: are distinct buffers. Copies and synchronization make backend boundaries explicit.", 22),
    ("04 / Raise the boundary cost", "A changed assumption changes placement", "$ placementlens compare --graph examples/fragmented.plg\n  --boundary-ms 5\n\n" + table("expensive-boundaries.txt"), "At a fixed cost of 5 ms per copy, CPU-only wins the modeled comparison.", 18),
    ("05 / Correctness and clear limits", "A runnable foundation for the next stage", "VERIFIED OUTPUT: [[1, 14], [7, 1]]\n\nInvalid graph rejected:\n" + (source / "rejection.txt").read_text().strip() + "\n\n337 core assertions + 16 CLI invocations\nAddressSanitizer / UndefinedBehaviorSanitizer checks pass\n\nNEXT: calibrated profiles, empirical oracle,\n      bounded real-backend validation", "Replay assembled from captured executable output. Silent preview; add your own narration for the final submission.", 18),
]
for i, (title, subtitle, content, caption, seconds) in enumerate(scenes):
    im = Image.new("RGB", (1600, 900), "#0d1620")
    d = ImageDraw.Draw(im)
    d.text((64, 36), "BOUNDARY CONDITIONS  /  EXPLAINABLE COMPILERS", font=font(19), fill="#8dabbc")
    d.text((64, 84), title, font=font(42), fill="#f0f7fb")
    d.text((64, 148), subtitle, font=font(25), fill="#53d5bd")
    d.rounded_rectangle((54, 209, 1546, 740), radius=12, fill="#152431", outline="#314b5d", width=2)
    lines = content.splitlines()
    text_size = min(25, 1330 // max(1, max(map(len, lines))) * 2 - 1)
    text_size = max(18, text_size)
    while max(d.textlength(line, font=font(text_size)) for line in lines) > 1410:
        text_size -= 1
    line_height = min(36, 485 // max(1, len(lines)))
    for j, line in enumerate(lines):
        d.text((82, 232 + j * line_height), line, font=font(text_size), fill="#dce8f0")
    for j, line in enumerate(textwrap.wrap(caption, width=110)):
        d.text((64, 778 + 31 * j), line, font=font(21), fill="#c1d2de")
    d.text((1410, 852), f"{i+1} / {len(scenes)}", font=font(18), fill="#8dabbc")
    im.save(frames / f"scene-{i+1}.png")
concat = frames / "frames.txt"
concat.write_text("".join(f"file 'scene-{i+1}.png'\nduration {scene[-1]}\n" for i, scene in enumerate(scenes)) + f"file 'scene-{len(scenes)}.png'\n")
output = source / "PlacementLens_Demo_Replay.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-vf", "fps=24", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
print(output)
