# embed_resource.cmake — Convert a file to a C++ byte array header
# Usage: cmake -DINPUT=file.html -DOUTPUT=embedded.h -DVAR=my_data -P embed_resource.cmake

file(READ "${INPUT}" content HEX)

# Split hex string into 0xNN bytes
string(LENGTH "${content}" hex_len)
set(bytes "")
set(i 0)
while(i LESS hex_len)
    string(SUBSTRING "${content}" ${i} 2 byte)
    if(bytes)
        string(APPEND bytes ",")
    endif()
    string(APPEND bytes "0x${byte}")
    math(EXPR i "${i} + 2")
    # Line break every 20 bytes
    math(EXPR col "${i} / 2")
    math(EXPR mod "${col} % 20")
    if(mod EQUAL 0 AND i LESS hex_len)
        string(APPEND bytes "\n    ")
    endif()
endwhile()

math(EXPR byte_count "${hex_len} / 2")

file(WRITE "${OUTPUT}"
    "// Auto-generated from ${INPUT} — do not edit\n"
    "#pragma once\n\n"
    "static const unsigned char ${VAR}_data[] = {\n    ${bytes}\n};\n"
    "static const unsigned int ${VAR}_size = ${byte_count};\n"
)
