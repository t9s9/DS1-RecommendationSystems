import src.frontend.algorithm
import src.frontend.menu
import src.frontend.reddit_dataset
import src.frontend.lol_dataset
from src.frontend.util import force_rerun


class PageHandler:
    def __init__(self, start):
        self.current_page = start
        self.pages = dict()

    def add_page(self, page, name):
        try:
            self.pages[name] = page.app
        except AttributeError:
            raise AttributeError(f"Cannot run page {name} because it has no app() function.")

    def set_page(self, page):
        if page in self.pages.keys():
            self.current_page = page
            force_rerun()
        else:
            raise ValueError(f"Page '{page}' not found.")

    def run(self):
        return self.pages[self.current_page]()


handler = PageHandler(start="als")
handler.add_page(src.frontend.menu, "menu")
handler.add_page(src.frontend.reddit_dataset, "reddit_dataset")
handler.add_page(src.frontend.algorithm, "als")
handler.add_page(src.frontend.lol_dataset, "lol_dataset")
