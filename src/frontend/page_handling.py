import src.frontend.menu
import src.frontend.reddit_dataset


class PageHandler:
    def __init__(self, start):
        self.current_page = start
        self.pages = dict()

    def add_page(self, page, name):
        self.pages[name] = page.app

    def set_page(self, page):
        if page in self.pages.keys():
            self.current_page = page
        else:
            raise ValueError(f"Page {page} not found.")

    def run(self):
        return self.pages[self.current_page]()


handler = PageHandler(start="menu")
handler.add_page(src.frontend.menu, "menu")
handler.add_page(src.frontend.reddit_dataset, "reddit_dataset")
print(handler.pages)
