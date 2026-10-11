class Tee:
    def __init__(self, console_stream, file_stream):
        self.console = console_stream
        self.file = file_stream
        self._file_line_buffer = ""

    def write(self, data):
        self.console.write(data)

        for char in data:
            if char == '\r':
                self._file_line_buffer = ""
            elif char == '\n':
                self.file.write(self._file_line_buffer + '\n')
                self._file_line_buffer = ""
            else:
                self._file_line_buffer += char

    def flush(self):
        self.console.flush()
        self.file.flush()
