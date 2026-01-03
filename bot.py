import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional

import discord
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TARGET_CHANNEL_ID = int(os.getenv("TARGET_CHANNEL_ID", "0"))

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

# ✅ Limite d'images analysées par message
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "2"))

FALLBACK_MSG = (
    "Slt {mention}, désolé je n'arrive pas à identifier l'article sur ta photo 😕 "
    "Renvoie une image plus nette (logo/étiquette bien visible) 🙏"
)

RATE_LIMIT_MSG = (
    "⏳ Slt {mention}, j’ai atteint temporairement la limite OpenAI (429). "
    "Réessaie dans 30–60 secondes 🙏"
)

SUCCESS_TEMPLATE = (
    "Slt {mention} ! D'après ce que je vois, tu aimerais une estimation pour **{item_name}**.\n"
    "À mon avis tu pourrais le revendre {estimate_txt} sur Vinted, à condition qu'il soit en très bon état 😉"
)

client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
bot = discord.Client(intents=intents)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# ✅ évite plusieurs appels OpenAI en même temps (réduit les 429)
ANALYZE_SEMAPHORE = asyncio.Semaphore(1)


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
    Analyse les images du message (limitées à MAX_IMAGES), renvoie UNE SEULE estimation globale.
    """
    instruction = (
        "Tu es un expert Vinted.\n"
        "Je t'envoie plusieurs photos du MÊME article (ex: photo générale + étiquette). "
        "Ton objectif est d'identifier au mieux l'article et de donner UNE estimation de revente Vinted.\n\n"
        "IMPORTANT:\n"
        "- Si tu n'es pas sûr, mets identified=false.\n"
        "- Donne un score confidence entre 0 et 1.\n"
        "- Réponds UNIQUEMENT en JSON valide.\n\n"
        "JSON attendu:\n"
        "{\n"
        '  "identified": true/false,\n'
        '  "confidence": 0.0,\n'
        '  "item_name": "string",\n'
        '  "price_range": [min,max] | null,\n'
        '  "suggested_price": number | null\n'
        "}\n"
    )

    content = [{"type": "input_text", "text": instruction}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url})

    async with ANALYZE_SEMAPHORE:
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
        return  # on ignore tout sauf les images

    # ✅ Ne garder que les 2 premières images (ou MAX_IMAGES)
    extra = len(image_urls) - MAX_IMAGES
    if extra > 0:
        image_urls = image_urls[:MAX_IMAGES]
        await message.reply(
            f"ℹ️ {message.author.mention} j’analyse uniquement les **{MAX_IMAGES} premières images** "
            f"(j’ai ignoré {extra} autre(s) image(s))."
        )

    # ✅ Une seule réponse par message
    try:
        result = await analyze_images(image_urls)
    except RateLimitError:
        await message.reply(RATE_LIMIT_MSG.format(mention=message.author.mention))
        return
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

    reply = SUCCESS_TEMPLATE.format(
        mention=message.author.mention,
        item_name=item_name,
        estimate_txt=estimate_txt,
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
