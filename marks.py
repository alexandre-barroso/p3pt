"""Remove acute and circumflex combining marks; keep all other marks."""
import unicodedata as ud
import model as sp
AC={'\u0301','\u0302'}
def selective(word):
    return ud.normalize('NFC',''.join(c for c in ud.normalize('NFD',sp.canonical(word)) if c not in AC))
