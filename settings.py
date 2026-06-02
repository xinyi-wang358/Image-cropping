from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qwen_api_key: str
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3-vl-plus"
    detect_confidence_threshold: float = 0.85
    target_width: int = 900
    target_height: int = 800
    slot_width: int = 300
    slot_height: int = 800
    target_waist_y: int = 80
    target_hem_y: int = 760
    annotate_margin_top_ratio: float = 0.04
    annotate_margin_bottom_ratio: float = 0.04
    annotate_margin_left_ratio: float = 0.05
    annotate_margin_right_ratio: float = 0.05
    dino_box_threshold: float = 0.38
    dino_text_threshold: float = 0.30
    dino_min_area_ratio: float = 0.02
    dino_max_area_ratio: float = 0.65
    dino_min_aspect_ratio: float = 0.18
    dino_max_aspect_ratio: float = 1.1
    dino_center_max_offset_ratio: float = 0.33

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
