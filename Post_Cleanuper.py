import re as regex

class Text_Cleanuper():
    def iterate(self):
        for cleanup_method in [self.remove_urls, self.remove_special_chars, self.remove_numbers, self.remove_english_words, self.to_lower]:
           yield cleanup_method

    @staticmethod
    def remove_by_regex(post, regexp):
        return regex.sub(regexp, '', post)

    def remove_urls(self, post):
        post = Text_Cleanuper.remove_by_regex(post, r'http.?://[^\s]+[\s]?')
        return post

    def remove_special_chars(self, post):
        return Text_Cleanuper.remove_by_regex(post, r'\`|\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\+|\=|\[|\{|\]|\}|\||\\|\'|\<|\,|\.|\>|\?|\/|\"')

    def remove_numbers(self, post):
        return Text_Cleanuper.remove_by_regex(post, r'\s?[0-9]+\.?[0-9]*')

    def remove_english_words(self, post):
        return Text_Cleanuper.remove_by_regex(post, r'\s?[a-zA-Z]+\.?[a-zA-Z]*')

    def to_lower(self, post):
        return post.lower()


class Posts_Cleansing(Text_Cleanuper):
    def __init__(self, data):
        self.processed_data = data.posts

    def cleanup(self, cleanuper):
        t = self.processed_data
        for cleanup_method in cleanuper.iterate():
            for idx, text in enumerate(t):
                    t[idx] = cleanup_method(text)

        self.processed_data = t
