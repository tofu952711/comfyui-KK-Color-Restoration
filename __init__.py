from .masked_color_transfer import KKColorRestore
from .scopes import KKDaVinciScopes
from .compare_scopes import KKDualImageColorScopes


WEB_DIRECTORY = "./web"


NODE_CLASS_MAPPINGS = {
    "KKColorRestore": KKColorRestore,
    "KKDaVinciScopes": KKDaVinciScopes,
    "KKDualImageColorScopes": KKDualImageColorScopes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KKColorRestore": "KK色彩还原",
    "KKDaVinciScopes": "KK达芬奇示波器",
    "KKDualImageColorScopes": "KK双图调色示波器",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
