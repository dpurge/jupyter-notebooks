from html import escape
from collections.abc import AsyncIterator
from typing import Any

from IPython.display import HTML, Markdown, display
import matplotlib.pyplot as plt
import cv2

async def render_response(events: AsyncIterator[dict[str, Any]]) -> Markdown:
    rendered_text = ""

    async for event in events:
        content = event.get("content", {})
        parts = content.get("parts", [])
        text_parts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ]
        if text_parts:
            rendered_text = "\n".join(text_parts).strip()

    return Markdown(rendered_text)


def render_user_message(message: str) -> HTML:
    escaped_message = escape(message).replace("\n", "<br>")
    return HTML(
        f"""
        <div style="
            margin: 1rem 0 0.75rem auto;
            max-width: 80%;
            padding: 0.85rem 1rem;
            border-radius: 16px 16px 4px 16px;
            background: linear-gradient(135deg, #0f766e, #115e59);
            color: #f8fafc;
            box-shadow: 0 10px 25px rgba(15, 118, 110, 0.18);
        ">
            <div style="
                margin-bottom: 0.35rem;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                opacity: 0.82;
            ">You</div>
            <div style="line-height: 1.5;">{escaped_message}</div>
        </div>
        """
    )

def display_text(text: str, style: str = None) -> HTML:
    if style:
        display(HTML(f"<pre style='{style}'>{text}</pre>"))
    else:
        display(HTML(f"<pre>{text}</pre>"))

def display_image(img, name='Image'):
    height, width, depth = img.shape
    scale = width / 15
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(height//scale, width//scale))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.show()