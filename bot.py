import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional

import discord
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TARGET_CHANNEL_ID = int(os.getenv("TARGET_CHANNEL_ID", "0"))

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

FALLBACK_MSG = (
    "Slt {mention}, désolé je n'arrive pas à identifier l'article sur ta photo 😕 "
    "Renvoie une image plus nette (logo/étiquette bien visible) 🙏"
)

client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
bot = discord.Client(intents=intents)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def is_image_attachment(att: discord.Attachment) -> bool:
    if att.content_type and att.content_type.startswith("image/"):
        return True
    filename = (att.filename or "").lower()
    return any(filename.endswith(ext) for ext in IMAGE_EXTS)


def extract_image_urls(message: discord.Message) -> List[str]:
    urls: List[str] = []

    for att in message.attachments:
        if is_image_attachment(att):
            urls.append(att.url)

    for emb in message.embeds:
        if emb.image and emb.image.url:
            urls.append(emb.image.url)
        if emb.thumbnail and emb.thumbnail.url:
            urls.append(emb.thumbnail.url)

    # dédoublonnage
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def analyze_images(image_urls: List[str]) -> Dict[str, Any]:
    """
    Analyse TOUTES les images du message (même article),
    mais renvoie UNE SEULE estimation + infos (marque, taille, etc).
    """
    instruction = (
        "Tu es un expert Vinted. Je t'envoie plusieurs photos du MÊME article "
        "(ex: photo globale + étiquette taille). Analyse-les ensemble.\n\n"
        "IMPORTANT:\n"
        "- Si tu n'es pas sûr, mets identified=false.\n"
        "- Donne confidence entre 0 et 1.\n"
        "- Si un champ n'est pas visible, mets null.\n"
        "- Réponds UNIQUEMENT en JSON valide.\n\n"
        "JSON attendu:\n"
        "{\n"
        '  "identified": true/false,\n'
        '  "confidence": 0.0,\n'
        '  "category": "string|null",\n'
        '  "brand": "string|null",\n'
        '  "model": "string|null",\n'
        '  "size": "string|null",\n'
        '  "color": "string|null",\n'
        '  "material": "string|null",\n'
        '  "item_name": "string|null",\n'
        '  "price_range": [min,max] | null,\n'
        '  "suggested_price": number | null,\n'
        '  "notes": "string|null"\n'
        "}\n"
    )

    content = [{"type": "input_text", "text": instruction}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url})

    resp = await asyncio.to_thread(
        client.responses.create,
        model=MODEL_NAME,
        input=[{
            "role": "user",
            "content": content,
        }],
    )

    out_text = getattr(resp, "output_text", "") or ""
    data = safe_parse_json(out_text) or {"identified": False, "confidence": 0.0}
    return data


def format_estimate_text(result: Dict[str, Any]) -> str:
    suggested = result.get("suggested_price")
    pr = result.get("price_range")

    if isinstance(suggested, (int, float)):
        return f"autour des **{int(suggested)}€**"
    if isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, (int, float)) for x in pr):
        return f"entre **{int(pr[0])}€** et **{int(pr[1])}€**"
    return "à un prix correct"


def build_details_line(result: Dict[str, Any]) -> str:
    parts = []
    for label, key in [
        ("Marque", "brand"),
        ("Modèle", "model"),
        ("Taille", "size"),
        ("Couleur", "color"),
        ("Matière", "material"),
    ]:
        val = result.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"**{label} :** {val.strip()}")
    return "\n".join(parts)


@bot.event
async def on_ready():
    print(f"✅ Connecté en tant que {bot.user} | salon surveillé: {TARGET_CHANNEL_ID}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.channel.id != TARGET_CHANNEL_ID:
        return

    image_urls = extract_image_urls(message)
    if not image_urls:
        return  # ignore tout sauf images

    # ✅ Une seule analyse + une seule réponse par message
    try:
        result = await analyze_images(image_urls)
    except Exception as e:
        print("OpenAI error:", e)
        await message.reply(FALLBACK_MSG.format(mention=message.author.mention))
        return

    identified = bool(result.get("identified", False))
    confidence = float(result.get("confidence", 0.0))

    if (not identified) or (confidence < CONFIDENCE_THRESHOLD):
        await message.reply(FALLBACK_MSG.format(mention=message.author.mention))
        return

    item_name = result.get("item_name") or "cet article"
    estimate_txt = format_estimate_text(result)
    details = build_details_line(result)

    notes = result.get("notes")
    notes_txt = f"\n\n💡 {notes.strip()}" if isinstance(notes, str) and notes.strip() else ""

    reply = (
        f"Slt {message.author.mention} ! D'après ce que je vois, tu aimerais une estimation pour **{item_name}**.\n"
        f"À mon avis tu pourrais le revendre {estimate_txt} sur Vinted, à condition qu'il soit en très bon état 😉\n\n"
        f"{details}"
        f"{notes_txt}"
    )

    await message.reply(reply)


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN manquant")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant")
    if TARGET_CHANNEL_ID == 0:
        raise RuntimeError("TARGET_CHANNEL_ID manquant")

    bot.run(DISCORD_BOT_TOKEN)
