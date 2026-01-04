import os
import asyncio
from typing import List, Dict, Any

import discord
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TARGET_CHANNEL_ID = int(os.getenv("TARGET_CHANNEL_ID", "0"))

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
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
    "Sur Vinted (seconde main), je dirais {estimate_txt}. "
    "Si l'article est en très bon état tu peux viser le haut de la fourchette 😉"
)

client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
bot = discord.Client(intents=intents)

# ✅ évite plusieurs appels OpenAI en même temps (réduit les 429)
ANALYZE_SEMAPHORE = asyncio.Semaphore(1)


def extract_image_urls(message: discord.Message) -> List[str]:
    urls: List[str] = []

    # Attachments
    for att in message.attachments:
        # content_type fiable si dispo
        if att.content_type and att.content_type.startswith("image/"):
            urls.append(att.url)
            continue
        # fallback via extension
        fn = (att.filename or "").lower()
        if fn.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            urls.append(att.url)

    # Embeds (images / thumbnails)
    for emb in message.embeds:
        if emb.image and emb.image.url:
            urls.append(emb.image.url)
        if emb.thumbnail and emb.thumbnail.url:
            urls.append(emb.thumbnail.url)

    # Dédoublonnage en conservant l'ordre
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def format_estimate_text(result: Dict[str, Any]) -> str:
    suggested = result.get("suggested_price")
    pr = result.get("price_range")

    if isinstance(suggested, (int, float)):
        return f"autour des **{int(suggested)}€**"
    if (
        isinstance(pr, list)
        and len(pr) == 2
        and all(isinstance(x, (int, float)) for x in pr)
    ):
        return f"entre **{int(pr[0])}€** et **{int(pr[1])}€**"
    return "à un prix cohérent"


async def analyze_images(image_urls: List[str]) -> Dict[str, Any]:
    """
    Analyse les images (limitées à MAX_IMAGES côté on_message) et renvoie UNE estimation globale.
    JSON garanti via json_schema.
    """
    instruction = (
        "Tu es un expert de la revente sur Vinted (seconde main) en France.\n"
        "Tu dois donner une estimation REALISTE de revente Vinted (prix auquel ça se vend), "
        "pas un prix neuf, pas un prix boutique.\n\n"
        "Règles IMPORTANTES (Vinted):\n"
        "- Par défaut, suppose un article D'OCCASION.\n"
        "- Sois conservateur: si tu hésites, baisse l'estimation.\n"
        "- Donne une fourchette: min = vente rapide, max = vente plus lente mais réaliste.\n"
        "- Si marque inconnue / peu demandée, reste bas.\n"
        "- Ne te base PAS sur le prix retail.\n\n"
        "Tu reçois plusieurs photos du MÊME article (photo globale + étiquette). Analyse-les ensemble.\n"
        "Réponds uniquement avec les champs demandés."
    )

    schema = {
        "name": "vinted_estimation",
        "schema": {
            "type": "object",
            "properties": {
                "identified": {"type": "boolean"},
                "confidence": {"type": "number"},
                "item_name": {"type": "string"},
                "price_range": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "suggested_price": {"type": ["number", "null"]},
            },
            "required": ["identified", "confidence", "item_name", "price_range", "suggested_price"],
            "additionalProperties": False
        }
    }

    content = [{"type": "input_text", "text": instruction}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url})

    async with ANALYZE_SEMAPHORE:
        resp = await asyncio.to_thread(
            client.responses.create,
            model=MODEL_NAME,
            response_format={"type": "json_schema", "json_schema": schema},
            input=[{"role": "user", "content": content}],
        )

    # JSON garanti par response_format => output_parsed
    data = getattr(resp, "output_parsed", None)
    if isinstance(data, dict):
        return data

    # fallback ultra safe (ne devrait pas arriver)
    return {"identified": False, "confidence": 0.0, "item_name": "cet article", "price_range": None, "suggested_price": None}


@bot.event
async def on_ready():
    print(f"✅ Connecté en tant que {bot.user} | salon surveillé: {TARGET_CHANNEL_ID} | max images: {MAX_IMAGES}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.channel.id != TARGET_CHANNEL_ID:
        return

    image_urls = extract_image_urls(message)
    if not image_urls:
        return  # on ignore tout sauf les images

    # ✅ Limiter aux MAX_IMAGES premières images
    extra = len(image_urls) - MAX_IMAGES
    if extra > 0:
        image_urls = image_urls[:MAX_IMAGES]
        await message.reply(
            f"ℹ️ {message.author.mention} j’analyse uniquement les **{MAX_IMAGES} premières images** "
            f"(j’ai ignoré {extra} autre(s) image(s))."
        )

    try:
        result = await analyze_images(image_urls)
    except RateLimitError:
        await message.reply(RATE_LIMIT_MSG.format(mention=message.author.mention))
        return
    except Exception as e:
        print("OpenAI error:", repr(e))
        await message.reply(FALLBACK_MSG.format(mention=message.author.mention))
        return

    identified = bool(result.get("identified", False))
    try:
        confidence = float(result.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if (not identified) or (confidence < CONFIDENCE_THRESHOLD):
        await message.reply(FALLBACK_MSG.format(mention=message.author.mention))
        return

    item_name = (result.get("item_name") or "cet article").strip()
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
