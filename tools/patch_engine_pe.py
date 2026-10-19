import struct
import os
import ctypes
import shutil
from ctypes import wintypes
import pefile

def patch_version(raw, new_desc):
    target = 'FileDescription\0'.encode('utf-16le')
    pos = raw.find(target)
    if pos == -1:
        return raw
    str_header_pos = pos - 6
    orig_wLen, orig_wValLen, orig_wType = struct.unpack_from('<HHH', raw, str_header_pos)
    
    new_val_str = new_desc + '\0'
    new_val_bytes = new_val_str.encode('utf-16le')
    new_val_wlen = len(new_val_str)
    
    old_entry_end = str_header_pos + orig_wLen
    if old_entry_end % 4 != 0:
        old_entry_end += (4 - (old_entry_end % 4))
    
    new_entry_header = bytearray(struct.pack('<HHH', 0, new_val_wlen, orig_wType) + target)
    if len(new_entry_header) % 4 != 0:
        new_entry_header += b'\x00' * (4 - (len(new_entry_header) % 4))
    new_entry = new_entry_header + new_val_bytes
    if len(new_entry) % 4 != 0:
        new_entry += b'\x00' * (4 - (len(new_entry) % 4))
    
    struct.pack_into('<H', new_entry, 0, len(new_entry))
    
    delta = len(new_entry) - (old_entry_end - str_header_pos)
    new_raw = bytearray(raw[:str_header_pos] + new_entry + raw[old_entry_end:])
    
    struct.pack_into('<H', new_raw, 0, len(new_raw))
    
    sfi_pos = new_raw.find('StringFileInfo\0'.encode('utf-16le'))
    if sfi_pos != -1:
        sfi_header = sfi_pos - 6
        sfi_len = struct.unpack_from('<H', new_raw, sfi_header)[0]
        struct.pack_into('<H', new_raw, sfi_header, sfi_len + delta)
        
        st_pos = sfi_pos + len('StringFileInfo\0'.encode('utf-16le'))
        if st_pos % 4 != 0:
            st_pos += (4 - (st_pos % 4))
        st_len = struct.unpack_from('<H', new_raw, st_pos)[0]
        struct.pack_into('<H', new_raw, st_pos, st_len + delta)
        
    return bytes(new_raw)

def patch_exe(exe_path, ico_path):
    exe_path = os.path.abspath(exe_path)
    if not os.path.exists(exe_path) or not os.path.exists(ico_path):
        return False

    with open(ico_path, 'rb') as f:
        ico_data = f.read()

    idReserved, idType, idCount = struct.unpack_from('<HHH', ico_data, 0)
    
    grp_dir = struct.pack('<HHH', idReserved, idType, idCount)
    entries = []
    offset = 6
    for i in range(idCount):
        bWidth, bHeight, bColorCount, bReserved, wPlanes, wBitCount, dwBytesInRes, dwImageOffset = struct.unpack_from('<BBBBHHII', ico_data, offset)
        offset += 16
        nID = i + 1
        grp_entry = struct.pack('<BBBBHHIH', bWidth, bHeight, bColorCount, bReserved, wPlanes, wBitCount, dwBytesInRes, nID)
        grp_dir += grp_entry
        entries.append((nID, dwBytesInRes, dwImageOffset))

    # Read original RT_VERSION
    pe = pefile.PE(exe_path)
    raw_ver = None
    ver_lang = 1033
    if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
        for rsrc in pe.DIRECTORY_ENTRY_RESOURCE.entries:
            if rsrc.id == pefile.RESOURCE_TYPE['RT_VERSION']:
                for entry in rsrc.directory.entries:
                    for item in entry.directory.entries:
                        data_rva = item.data.struct.OffsetToData
                        size = item.data.struct.Size
                        raw_ver = pe.get_data(data_rva, size)
                        ver_lang = item.id
    pe.close()

    new_ver = patch_version(raw_ver, "SpektraDarkroom") if raw_ver else None

    kernel32 = ctypes.windll.kernel32
    kernel32.BeginUpdateResourceW.restype = wintypes.HANDLE
    kernel32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]

    kernel32.UpdateResourceW.restype = wintypes.BOOL
    kernel32.UpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.WORD, wintypes.LPVOID, wintypes.DWORD]

    kernel32.EndUpdateResourceW.restype = wintypes.BOOL
    kernel32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]

    h = kernel32.BeginUpdateResourceW(exe_path, False)
    if not h:
        print(f"BeginUpdateResourceW failed on {exe_path}: {ctypes.GetLastError()}")
        return False

    RT_ICON = wintypes.LPCWSTR(3)
    RT_GROUP_ICON = wintypes.LPCWSTR(14)
    RT_VERSION = wintypes.LPCWSTR(16)

    for nID, size, img_offset in entries:
        raw_icon = ico_data[img_offset:img_offset+size]
        kernel32.UpdateResourceW(h, RT_ICON, wintypes.LPCWSTR(nID), 0, raw_icon, len(raw_icon))

    kernel32.UpdateResourceW(h, RT_GROUP_ICON, wintypes.LPCWSTR(1), 0, grp_dir, len(grp_dir))

    if new_ver:
        kernel32.UpdateResourceW(h, RT_VERSION, wintypes.LPCWSTR(1), ver_lang, new_ver, len(new_ver))

    ok = kernel32.EndUpdateResourceW(h, False)
    print(f"Patched {exe_path}: success={bool(ok)}")
    return bool(ok)

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    ico = os.path.join(base, "resources", "app_icon.ico")
    
    # 1. .venv/Scripts/SpektraDarkroomEngine.exe
    venv_py = os.path.join(base, ".venv", "Scripts", "python.exe")
    venv_engine = os.path.join(base, ".venv", "Scripts", "SpektraDarkroomEngine.exe")
    if os.path.exists(venv_py):
        try:
            shutil.copyfile(venv_py, venv_engine)
            patch_exe(venv_engine, ico)
        except Exception as e:
            print("Failed to patch venv engine:", e)

    # 2. runtime/SpektraDarkroomEngine.exe if runtime exists
    rt_py = os.path.join(base, "runtime", "python.exe")
    rt_engine = os.path.join(base, "runtime", "SpektraDarkroomEngine.exe")
    if os.path.exists(rt_py):
        try:
            shutil.copyfile(rt_py, rt_engine)
            patch_exe(rt_engine, ico)
        except Exception as e:
            print("Failed to patch runtime engine:", e)
