from book import ContentType

class Model:
    def make_text_prompt(self, text: str, target_language: str) -> str:
        return f"把下面第一个：后面的所有文本翻译为{target_language}，只允许输出翻译的结果：\n{text}"

    def make_table_prompt(self, table: str, target_language: str) -> str:
        # return f"翻译为{target_language}，保持间距（空格，分隔符），以表格形式返回：\n{table}"
        return f"把下面第一个：后面的所有文本翻译为{target_language}。只允许按照格式输出翻译的结果，以空格和换行符表示输出表格的格式。列与列之间会使用空格进行分割，必须确保输出中的列数相同，必要时可以使用-进行填充：\n{table}"

    def translate_prompt(self, content, target_language: str) -> str:
        if content.content_type == ContentType.TEXT:
            return self.make_text_prompt(content.original, target_language)
        elif content.content_type == ContentType.TABLE:
            return self.make_table_prompt(content.get_original_as_str(), target_language)

    def make_request(self, prompt):
        raise NotImplementedError("子类必须实现 make_request 方法")
