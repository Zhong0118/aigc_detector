"""Read bounded ZIP members through public HTTP ranges, including split volumes.

ZIP directory parsing uses Python's zipfile. Only stored/deflated, non-encrypted,
non-ZIP64 archives are supported. CRC and decoded size are checked per member;
partial downloads cannot verify the publisher's whole-volume MD5.
"""
import io
import struct
import zipfile
import zlib


def archive_catalog(sizes, fetch):
    disk = len(sizes)-1
    tail = fetch(disk, max(0, sizes[disk]-65557), min(sizes[disk], 65557))
    end = tail.rfind(b'PK\x05\x06')
    if end < 0: raise ValueError('ZIP end record missing')
    sig, current, central_disk, local_n, total_n, cd_size, cd_offset, comment_n = struct.unpack_from('<4s4H2IH', tail, end)
    if current < disk or (current != disk and disk != 0) or central_disk != current or total_n == 65535 or cd_size > 8*1024*1024:
        raise ValueError('unsupported ZIP disk/ZIP64 layout')
    if end+22+comment_n != len(tail) or cd_offset+cd_size != sizes[disk]-(len(tail)-end):
        raise ValueError('inconsistent ZIP central directory bounds')
    central = fetch(disk, cd_offset, cd_size)
    with zipfile.ZipFile(io.BytesIO(central+tail[end:])) as z:
        if len(z.infolist()) != total_n: raise ValueError('ZIP entry count mismatch')
        entries=[]
        for info in z.infolist():
            if info.is_dir(): continue
            entries.append(dict(filename=info.filename, volume=info.volume,
                physical_volume=info.volume-(current-disk), archive_last_volume=current,
                offset=info.header_offset+cd_offset, compressed_size=info.compress_size,
                size=info.file_size, crc32=info.CRC, compression=info.compress_type,
                flags=info.flag_bits))
    return entries


def read_span(sizes, fetch, disk, start, length):
    chunks=[]
    while length:
        if disk >= len(sizes) or start < 0 or start > sizes[disk]:
            raise ValueError('ZIP member outside volume bounds')
        n=min(length,sizes[disk]-start)
        if n:
            data=fetch(disk,start,n)
            if len(data)!=n: raise ValueError('short range response')
            chunks.append(data);length-=n
        disk+=1;start=0
    return b''.join(chunks)


def read_member(entry, sizes, fetch):
    if entry['flags'] & 1 or entry['compression'] not in (0,8):
        raise ValueError('encrypted or unsupported ZIP compression')
    if not 0 <= entry['size'] <= 50*1024*1024 or entry['compressed_size'] > 50*1024*1024:
        raise ValueError('oversize ZIP member')
    disk,start=entry.get('physical_volume',entry['volume']),entry['offset']
    if disk < 0: raise ValueError('ZIP member starts in unavailable volume')
    remaining=sum(sizes[disk:])-start
    raw=read_span(sizes,fetch,disk,start,min(remaining,entry['compressed_size']+4096))
    if raw[:4]!=b'PK\x03\x04': raise ValueError('ZIP local header missing')
    n,e=struct.unpack_from('<HH',raw,26)
    header=30+n+e
    if header>4096: raise ValueError('unsupported large ZIP local header')
    name=raw[30:30+n].decode('utf-8' if entry['flags'] & 2048 else 'cp437')
    if name != entry['filename']: raise ValueError('ZIP local/central filename mismatch')
    packed=raw[header:header+entry['compressed_size']]
    if len(packed)!=entry['compressed_size']: raise ValueError('truncated ZIP member')
    if entry['compression']==8:
        inflater=zlib.decompressobj(-15)
        data=inflater.decompress(packed,entry['size']+1)
        if not inflater.eof or inflater.unconsumed_tail: raise ValueError('invalid deflate stream')
    else: data=packed
    if len(data)!=entry['size'] or zlib.crc32(data)!=entry['crc32']:
        raise ValueError('ZIP member size/CRC mismatch')
    return data
