from cleo.config import ApplicationConfig as BaseApplicationConfig
from clikit.api.formatter import Style


# ============================================
#              ApplicationConfig
# ============================================
class ApplicationConfig(BaseApplicationConfig):
    def configure(self) -> None:
        super().configure()
        self.add_style(Style("info").fg("cyan"))
        self.add_style(Style("error").fg("red").bold())
        self.add_style(Style("warning").fg("yellow").bold())
        self.add_style(Style("success").fg("green"))
