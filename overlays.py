"""On-screen overlays for reels, as ASS events (PlayRes 1080x1920).

Driven by optional fields on a clip in clips.json:
  headline      news card shown for the first `headline_secs` (default 3) seconds
  kicker        small red tag above the headline (default "NEWS")
  label         title bar pinned at the top for the rest of the clip
  source        credit line at the bottom, e.g. "Source: E! News"
  story         [{start, end, text}] big centred text, for footage without speech
"""
import re

STYLES = r"""Style: Head,Arial Black,96,&H000A0A0A,&H000A0A0A,&H0000E6FF,&H0000E6FF,-1,0,0,0,100,100,0,0,3,22,0,8,40,40,0,1
Style: Tag,Arial Black,46,&H00FFFFFF,&H00FFFFFF,&H001E1EE6,&H001E1EE6,-1,0,0,0,100,100,2,0,3,12,0,8,40,40,0,1
Style: Bar,Arial Black,54,&H0000E6FF,&H0000E6FF,&H50000000,&H50000000,-1,0,0,0,100,100,0,0,3,16,0,8,40,40,0,1
Style: Credit,Arial,30,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,40,40,0,1
Style: Story,Arial Black,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,5,60,60,0,1
"""


def _t(t: float) -> str:
    cs = int(round(max(t, 0) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _clean(text: str) -> str:
    return re.sub(r"[{}\\]", "", text)


def _wrap(text: str, width: int) -> str:
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\\N".join(lines)


def caption_start(clip: dict) -> float:
    """Spoken captions are held back until the headline card has gone."""
    return float(clip.get("headline_secs", 3.0)) if clip.get("headline") else 0.0


def events(clip: dict, dur: float) -> list[str]:
    ev = []
    end = _t(dur + 1)
    head_end = caption_start(clip)
    if clip.get("headline"):
        kicker = _clean(clip.get("kicker", "NEWS")).upper()
        head = _wrap(_clean(clip["headline"]).upper(), 15)
        n_lines = head.count("\\N") + 1
        # Lower third, growing upward from just above the credit, so faces stay
        # visible and long headlines never run off the bottom.
        bottom = 1760
        tag_y = bottom - n_lines * 122 - 30
        ev.append(f"Dialogue: 2,{_t(0)},{_t(head_end)},Tag,,0,0,0,,"
                  f"{{\\an2\\pos(540,{tag_y})\\fad(0,200)}}" + kicker)
        ev.append(f"Dialogue: 2,{_t(0)},{_t(head_end)},Head,,0,0,0,,"
                  f"{{\\an2\\pos(540,{bottom})\\fad(120,200)}}" + head)
    if clip.get("label"):
        ev.append(f"Dialogue: 1,{_t(head_end)},{end},Bar,,0,0,0,,"
                  "{\\an8\\pos(540,150)\\fad(200,0)}" + _wrap(_clean(clip["label"]), 26))
    if clip.get("source"):
        ev.append(f"Dialogue: 1,{_t(0)},{end},Credit,,0,0,0,,"
                  "{\\an2\\pos(540,1870)}" + _wrap(_clean(clip["source"]), 52))
    for st in clip.get("story", []):
        ev.append(f"Dialogue: 1,{_t(st['start'])},{_t(st['end'])},Story,,0,0,0,,"
                  "{\\an5\\pos(540,1240)\\fad(150,150)}" + _wrap(_clean(st["text"]), 20))
    return ev
