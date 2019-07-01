from .bot.en_fqa_bot    import EnglishFQABot
from .bot.vi_fqa_bot    import VietnameseFQABot

__all__ = (
    'VietnameseFQABot',
    'EnglishFQABot',
)

fqa_bots_map = {
    'vi' : VietnameseFQABot,
    'en' : EnglishFQABot
}

def get_fqa_bot(lang, **kwargs):

    return fqa_bots_map[lang](**kwargs)
