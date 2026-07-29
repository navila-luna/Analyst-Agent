from sqlalchemy.orm import Session

from app.models import BotConfig

CONFIG_ID = 1  # single active config for now - see PLAN.md Phase 5 note


def get_or_create_config(db: Session) -> BotConfig:
    config = db.get(BotConfig, CONFIG_ID)
    if config is None:
        config = BotConfig(id=CONFIG_ID)
        db.add(config)
        db.commit()
        db.refresh(config)
    return config
