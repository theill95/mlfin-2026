"""Stamp the temporary notice after page generators. Set ENABLED=False to remove it."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENABLED = True
START = '<!-- exam-notice:start -->'
END = '<!-- exam-notice:end -->'

def main():
    notice = (ROOT / 'assets/exam-notice.html').read_text(encoding='utf-8').strip()
    count = 0
    for path in ROOT.glob('*.html'):
        old = path.read_text(encoding='utf-8')
        text = re.sub(r'\n?' + re.escape(START) + r'.*?' + re.escape(END) + r'\n?', '\n', old, flags=re.S)
        # Exam info already has the expanded, permanent box first in its main content.
        if ENABLED and path.name != 'exam_info.html':
            text = re.sub(r'(<main\b[^>]*>)', lambda m: m[0] + '\n' + START + '\n' + notice + '\n' + END, text, count=1)
        if text != old:
            path.write_text(text, encoding='utf-8', newline='\n')
            count += 1
    print(f'Exam notice enabled={ENABLED}: {count} pages updated')

if __name__ == '__main__':
    main()
