"""Inline docs/dashboard_data.json into docs/dashboard_template.html -> docs/dashboard.html (one self-contained page)."""
from pathlib import Path
data = Path('docs/dashboard_data.json').read_text(encoding='utf-8').replace('</', '<\/')
html = Path('docs/dashboard_template.html').read_text(encoding='utf-8').replace('__DATA__', data)
Path('docs/dashboard.html').write_text(html, encoding='utf-8')
print('wrote docs/dashboard.html', round(len(html.encode('utf-8')) / 1024), 'KB')
