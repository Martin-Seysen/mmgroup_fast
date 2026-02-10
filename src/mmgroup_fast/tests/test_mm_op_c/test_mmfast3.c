// Warning: This file has been generated automatically. Do not change!

/// @cond DO_NOT_DOCUMENT 
#define DISPLAY_DLL_EXPORTS 
/// @endcond

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <time.h>

#include "mm_op_fast_types.h"
#include "mm_op_fast.h"
#include "mmgroup_fast_display.h"


#define MAX_LINE_LEN 4096
#define MAX_NUMS     1024




int parse_uint32(const char *str, uint32_t *value, size_t *chars_parsed) {
    // Parse hexadecimal or decimal integer from str, 
    // skipping leading whitespace.
    // parsed integer -> *value
    // No of characters parsed -> *chars_parsed
    // Return value:
    //  1 = an integer has been parsed
    //  0 = end of string
    // -1 = illegal character in str
    // -2 = overflow

    const char *p = str;
    char *endptr = NULL;
    unsigned long result;

    // Skip leading whitespace
    while (isspace((unsigned char)*p)) p++;
    if (*p == '\0') {
        *chars_parsed = (size_t)(p - str);
        return 0;
    }

    // Detect base
    int base = 10;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        base = 16; p += 2;
    }

    // Use strtoul for conversion
    result = strtoul(p, &endptr, base);

    // Check if any digits were parsed
    if (endptr == p) {
        return -1;  // no valid number found
    }

    // Check for overflow
    if (result > UINT32_MAX) {
        return -2;  // overflow
    }

    *value = (uint32_t)result;
    *chars_parsed = (size_t)(endptr - str);
    return 1; // success
}





int parse_int_line(char *line, unsigned int *out, int max_out) {
    int count = 0;
    char *ptr = line;
    if (!isspace(*ptr)) return -1; // leading blank expected
    while (*ptr != '\0' && count < max_out) {
        int status;
        size_t chars_parsed;
        status = parse_uint32(ptr, &out[count], &chars_parsed);
        if (status < 0) return status;
        if (status == 0) return count;
        ++count;
        ptr += chars_parsed;
    }

    return count;
}


uint32_t arg_to_uint32(char *str, uint32_t default_value) {  
    int status;
    uint32_t value;
    size_t chars_parsed;
    if (str == NULL) return default_value;
    status = parse_uint32(str, &value, &chars_parsed);
    return status == 1 ? value : default_value;
}


#define MAX_SLACK  1024


static volatile uint32_t align = 0, rand_sum = 0;


int32_t call_mm_op_fast_word(
    mmv_fast_matrix_type *pv, uint32_t *a, uint32_t len
)
{
#if defined(__GNUC__) || defined(__clang__)
    #define my_alloca(size)  __builtin_alloca(size)
#elif defined(_MSC_VER)
    #define my_alloca(size) _alloca(size)
#else
#   error "alloca_aligned is not supported on this compiler."
#endif

    volatile uint8_t *pbuf;
    int32_t status, i;
    // Allcate ``align`` bytes on the stack and perform dummy operation
    // to prevent compiler optimization 
    if (align) {
        pbuf = my_alloca(align);
        for (i = 0; i < align; ++i) pbuf[i] = (uint8_t)(rand() & 0xff);
    }
    // Do the real job
    status = mm_op_fast_word(pv, a, len, 1);
    // Continue dummy operation
    if (align) {
        for (i = 0; i < align; ++i) rand_sum += pbuf[i];
    }
    return status;
}


int work(char *f_in_name, char *f_out_name, uint32_t align_offset)
{
    static char line[MAX_LINE_LEN+2], tag;
    static uint32_t a[MAX_NUMS], i;
    static struct {
        mmv_fast_matrix_type v;
        char slack[MAX_SLACK];
    } vbuf;
    if (align_offset > MAX_SLACK) align_offset = 0;
    mmv_fast_matrix_type *pv 
        = (mmv_fast_matrix_type*)((char*)(&vbuf) + align_offset);
    uint8_t *pb;
    FILE *f_in, *f_out;
    int32_t status = 0, len;
   
    mm_op_fast_init(pv, 3, 4, 1);
    pb = mm_op_fast_raw_vb_data(pv, NULL);
    srand((unsigned int)time(NULL)); // seed with current time
    for (i = 0; i < MM_FAST_BYTELENGTH; ++i) {
        pb[i] = (uint8_t)((rand() & 0x55) + 0x55);
    }
    
    f_in = fopen(f_in_name, "rt");
    if (!f_in) {perror("fopen_in"); return 1;}
    f_out = fopen(f_out_name, "wt");
    if (!f_out) {perror("fopen_out"); return 2;}

    while (1) {
       status = 3;
       if (!fgets(line, sizeof(line), f_in)) {perror("read"); return 3;}
       fputs(line, f_out);
       fflush(f_out);
       status = 0;  
       switch(line[0]) {
           case 'E':
               goto done;
           case 'R':
               len = status = parse_int_line(line+1, a, MAX_NUMS); 
               if (status < 0) goto done;
               if (len == 1) a[1] = 1;
               status = 11;
               if (len < 1 || len > 2) goto done;
               status = 12;
               if ((uint64_t)a[0] + (uint64_t)a[1] >
                   (uint64_t)MM_FAST_BYTELENGTH) goto done;
               pb = mm_op_fast_raw_vb_data(pv, NULL) + a[0];
               fprintf(f_out, ">R");
               for (i = 0; i < a[1]; ++i) fprintf(f_out, " 0x%02x", pb[i]);
               fprintf(f_out,"\n");
               break;
           case 'M':
               status = len = parse_int_line(line+1, a, MAX_NUMS);   
               if (status < 0) goto done;
               status =  mm_op_fast_word(pv, a, len, 1);
               fprintf(f_out, ">M ok\n");
               break;
           case 'P':
               pb = mm_op_fast_raw_vb_data(pv, NULL);
               fprintf(f_out, ">P v: %p, buf: %p\n", pv, pb);
               break;
           case 'A':
               len = parse_int_line(line+1, a, MAX_NUMS);   
               if (len > 0 && 0 <= a[0] <= 256) align = a[0];
               fprintf(f_out, ">A %d", rand_sum);
               fprintf(f_out, ">M ok\n");
               break;
          default:
               status = 99;
               if (!isalpha(line[0])) line[0] = '?';
               goto done;
       }
       fflush(f_out);
    }
done:
    if (status) {
        fprintf(f_out, ">Error %d in operation %c\n", status, line[0]);
    }
    else {
        fprintf(f_out, ">Success\n");
    }
    fclose(f_out);
    fclose(f_in);
    return status ? 4 : 0;
}



int main(int argc, char **argv)
{
    if (argc < 3) return 5;
    uint32_t align_offset 
        = arg_to_uint32(argc > 3 ? argv[3] : 0, 0);
    return work(argv[1], argv[2], align_offset);
}

