#include <sionflow/host/sf_host_desc.h>
#include <sionflow/engine/sf_engine.h>
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_utils.h>
#include <sionflow/base/sf_shape.h>
#include "sf_host_internal.h"
#include "sf_loader.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

bool sf_loader_load_image(sf_engine* engine, const char* name, const char* path) {
    sf_tensor* t = sf_engine_map_resource(engine, name);
    if (!t) return false;

    unsigned char* data = NULL;
    int w, h, c, d = 0;
    if (t->info.ndim >= 3) d = t->info.shape[t->info.ndim - 1];

    // Try loading from cartridge first
    const char* ext = sf_path_get_ext(path);
    if (strcmp(ext, "sfc") == 0 || strcmp(ext, "bin") == 0) {
        sf_cartridge* cart = sf_cartridge_open(path);
        if (cart) {
            size_t section_size = 0;
            void* section_data = sf_cartridge_get_section(cart, name, SF_SECTION_IMAGE, &section_size);
            if (section_data) {
                data = stbi_load_from_memory(section_data, (int)section_size, &w, &h, &c, d);
                if (data) SF_LOG_INFO("Loaded embedded image '%s' from cartridge.", name);
            }
            sf_cartridge_close(cart);
        }
    }

    // Fallback to filesystem
    if (!data) {
        data = stbi_load(path, &w, &h, &c, d);
    }

    if (!data) return false;
    if (d == 0) d = c;
    int32_t sh[3]; uint8_t n = 0;
    if (d > 1) { sh[0] = h; sh[1] = w; sh[2] = d; n = 3; } else { sh[0] = h; sh[1] = w; n = 2; }
    if (!sf_engine_resize_resource(engine, name, sh, n)) { 
        SF_LOG_ERROR("Assets: Failed to resize resource '%s' for image loading.", name);
        stbi_image_free(data); 
        return false; 
    }
    
    t = sf_engine_map_resource(engine, name);
    if (!t || !t->buffer || !t->buffer->data) {
        SF_LOG_ERROR("Assets: Resource '%s' disappeared after resize.", name);
        stbi_image_free(data);
        return false;
    }

    size_t p = (size_t)w * h * d;
    size_t max_bytes = sf_tensor_size_bytes(t);
    
    if (t->info.dtype == SF_DTYPE_F32) { 
        if (max_bytes < p * sizeof(f32)) {
            SF_LOG_ERROR("Assets: Resource '%s' is too small for F32 image data.", name);
            stbi_image_free(data); return false;
        }
        f32* dst = (f32*)t->buffer->data; 
        for (size_t i = 0; i < p; ++i) dst[i] = (f32)data[i] / 255.0f; 
    }
    else if (t->info.dtype == SF_DTYPE_U8) {
        if (max_bytes < p) {
            SF_LOG_ERROR("Assets: Resource '%s' is too small for U8 image data.", name);
            stbi_image_free(data); return false;
        }
        memcpy(t->buffer->data, data, p);
    }
    else {
        SF_LOG_ERROR("Assets: Resource '%s' has unsupported dtype for image loading.", name);
        stbi_image_free(data); return false;
    }
    
    stbi_image_free(data); 
    sf_engine_sync_resource(engine, name); 
    return true;
}

static bool _bake_sdf(stbtt_fontinfo* f, int s, int e, u8* a, int aw, int ah, int* cx, int* cy, int l, f32* inf, int* c, float sc, int p, u8 edge, float dist) {
    for (int cp = s; cp < e; ++cp) {
        int g = stbtt_FindGlyphIndex(f, cp); if (g == 0) continue;
        int adv, lsb, gw, gh, xo, yo; stbtt_GetGlyphHMetrics(f, g, &adv, &lsb);
        u8* sdf = stbtt_GetGlyphSDF(f, sc, g, p, edge, dist, &gw, &gh, &xo, &yo); if (!sdf) continue;
        if (*cx + gw >= aw) { *cx = 0; *cy += l; } if (*cy + gh >= ah) { stbtt_FreeSDF(sdf, NULL); return false; }
        for (int y = 0; y < gh; ++y) memcpy(a + (*cy + y) * aw + *cx, sdf + y * gw, gw); stbtt_FreeSDF(sdf, NULL);
        int i = cp * 8; inf[i+0]=(f32)cp; inf[i+1]=(f32)*cx/aw; inf[i+2]=(f32)*cy/ah; inf[i+3]=(f32)(*cx+gw)/aw; inf[i+4]=(f32)(*cy+gh)/ah; inf[i+5]=(f32)adv*sc; inf[i+6]=(f32)xo; inf[i+7]=(f32)yo;
        (*c)++; *cx += gw + 1;
    }
    return true;
}

bool sf_loader_load_font(sf_engine* engine, const char* name, const char* path, float size) {
    // Font Baking Config
    const int atlas_w = 1024;
    const int atlas_h = 1024;
    const int max_glyphs = 2048;
    const int padding = 2;
    const float sdf_dist = 32.0f;
    const u8 sdf_edge = 128;

    size_t len = 0; 
    unsigned char* ttf = NULL;
    bool ttf_owned = false;
    
    // ... (cartridge loading logic) ...
    if (strcmp(ext, "sfc") == 0 || strcmp(ext, "bin") == 0) {
        sf_cartridge* cart = sf_cartridge_open(path);
        if (cart) {
            size_t section_size = 0;
            void* section_data = sf_cartridge_get_section(cart, name, SF_SECTION_FONT, &section_size);
            if (section_data) {
                ttf = malloc(section_size);
                memcpy(ttf, section_data, section_size);
                len = section_size;
                ttf_owned = true;
                SF_LOG_INFO("Loaded embedded font '%s' from cartridge.", name);
            }
            sf_cartridge_close(cart);
        }
    }

    if (!ttf) {
        ttf = (unsigned char*)sf_file_read_bin(path, &len);
        ttf_owned = true;
    }

    if (!ttf) return false;
    
    stbtt_fontinfo f; 
    if (!stbtt_InitFont(&f, ttf, 0)) { if (ttf_owned) free(ttf); return false; }
    
    float sc = stbtt_ScaleForPixelHeight(&f, size);
    u8* a = calloc(1, atlas_w * atlas_h);
    f32* inf = calloc(max_glyphs * 8, sizeof(f32));
    int ct = 0, cx = 0, cy = 0, cell = (int)(size * 1.5f);
    
    bool ok = true;
    // ASCII
    ok &= _bake_sdf(&f, 32, 127, a, atlas_w, atlas_h, &cx, &cy, cell, inf, &ct, sc, padding, sdf_edge, sdf_dist);
    // Cyrillic
    ok &= _bake_sdf(&f, 1024, 1104, a, atlas_w, atlas_h, &cx, &cy, cell, inf, &ct, sc, padding, sdf_edge, sdf_dist);

    if (!ok) {
        SF_LOG_ERROR("Assets: Font atlas overflow for '%s'.", name);
    }
    
    int32_t sh[] = { atlas_h, atlas_w }; 
    if (sf_engine_resize_resource(engine, name, sh, 2)) {
        sf_tensor* t = sf_engine_map_resource(engine, name); 
        if (t && t->buffer && t->buffer->data && t->info.dtype == SF_DTYPE_F32) {
            for(size_t i=0; i<(size_t)atlas_w*atlas_h; ++i) ((f32*)t->buffer->data)[i] = (f32)a[i] / 255.0f;
            sf_engine_sync_resource(engine, name);
        } else {
            SF_LOG_ERROR("Assets: Font resource '%s' must be F32.", name);
        }
    }
    
    char in[128]; 
    snprintf(in, 128, "%s_Info", name);
    int32_t ish[] = { max_glyphs * 8 }; 
    if (sf_engine_resize_resource(engine, in, ish, 1)) {
        sf_tensor* ti = sf_engine_map_resource(engine, in); 
        if (ti && ti->buffer && ti->buffer->data) {
             size_t max_bytes = sf_tensor_size_bytes(ti);
             size_t needed = max_glyphs * 8 * sizeof(f32);
             if (max_bytes >= needed) {
                 memcpy(ti->buffer->data, inf, needed);
                 sf_engine_sync_resource(engine, in);
             } else {
                 SF_LOG_ERROR("Assets: Font info resource '%s' is too small.", in);
             }
        }
    }
    
    free(a); free(inf); if (ttf_owned) free(ttf); 
    return true;
}
    
    free(a); free(inf); if (ttf_owned) free(ttf); 
    return true;
}
