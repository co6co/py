from model.apphelp import *
import pytest

@pytest.mark.asyncio
def test_read_file_content():
    html_content =  read_file_content("index.html")
    assert html_content != "404 Not Found: index.html"
@pytest.mark.asyncio
async def test_read_file_chunk():
    html_content = read_chunked("index.html",512)
    async for chunk in html_content:
        print(f"chunk {len(chunk)}->",chunk.decode()) 
@pytest.mark.asyncio
async def test_read_file_chunk_line():
    line_generator =    read_file_line("index.html")
    index=0
    async for line in line_generator:
        index+=1
        print(f"line {index}->",line) 
   