import io
import struct
import zipfile
import pytest
from research.image_attribution.iab_archives import archive_catalog, read_member


def fixture_parts():
    stream = io.BytesIO()
    expected = {'a.png': b'a'*2000, 'b.png': b'b'*3000, 'c.png': b'c'*300}
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as z:
        for name, data in expected.items(): z.writestr(name, data)
    raw = bytearray(stream.getvalue())
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        boundary = z.getinfo('b.png').header_offset + 100
        central = z.start_dir
    p = central
    while raw[p:p+4] == b'PK\x01\x02':
        n, e, c = struct.unpack_from('<HHH', raw, p+28)
        offset = struct.unpack_from('<I', raw, p+42)[0]
        struct.pack_into('<H', raw, p+34, int(offset >= boundary))
        struct.pack_into('<I', raw, p+42, offset-boundary if offset >= boundary else offset)
        p += 46+n+e+c
    struct.pack_into('<HH', raw, p+4, 1, 1)
    struct.pack_into('<I', raw, p+16, central-boundary)
    return [bytes(raw[:boundary]), bytes(raw[boundary:])], expected


def test_split_zip_reads_member_crossing_volume_boundary():
    parts, expected = fixture_parts()
    fetch = lambda disk, start, length: parts[disk][start:start+length]
    sizes = list(map(len, parts))
    entries = archive_catalog(sizes, fetch)
    assert {e['filename']:read_member(e, sizes, fetch) for e in entries} == expected


def test_crc_corruption_rejected():
    parts, _ = fixture_parts()
    fetch = lambda disk, start, length: parts[disk][start:start+length]
    entries = archive_catalog(list(map(len, parts)), fetch)
    parts[0] = parts[0][:100] + b'x' + parts[0][101:]
    with pytest.raises(ValueError, match='CRC'):
        read_member(entries[0], list(map(len, parts)), fetch)


def test_single_zip_deflate_and_paths_are_not_extracted():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('../x.png',b'hello'*1000)
    raw=stream.getvalue(); fetch=lambda disk,start,length:raw[start:start+length]
    entry=archive_catalog([len(raw)],fetch)[0]
    assert read_member(entry,[len(raw)],fetch)==b'hello'*1000


def test_last_volume_can_be_audited_without_missing_prefix():
    parts,expected=fixture_parts()
    last=parts[-1];fetch=lambda disk,start,length:last[start:start+length]
    entries=archive_catalog([len(last)],fetch)
    available=[e for e in entries if e['physical_volume']>=0]
    assert [e['filename'] for e in available]==['c.png']
    assert read_member(available[0],[len(last)],fetch)==expected['c.png']
    with pytest.raises(ValueError,match='unavailable volume'):
        read_member(entries[0],[len(last)],fetch)
