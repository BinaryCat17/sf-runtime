#include <sionflow/host/sf_host_sdl.h>
#include <sionflow/engine/sf_engine.h>
#include <sionflow/base/sf_platform.h>
#include <sionflow/base/sf_log.h>
#include "sf_host_internal.h"
#include "sf_loader.h"

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void convert_to_pixels(sf_tensor* tensor, void* pixels, int pitch, int tex_w, int tex_h) {
    (void)pitch;
    void* data_ptr = sf_tensor_data(tensor);
    if (!tensor || !data_ptr) return;
    
    f32* src = (f32*)data_ptr;
    u8* dst = (u8*)pixels;
    int total_pixels = tex_w * tex_h;
    int channels = tensor->info.ndim >= 3 ? tensor->info.shape[tensor->info.ndim - 1] : 1;

    for (int i = 0; i < total_pixels; ++i) {
        float r, g, b, a;
        if (channels >= 4) { r = src[i*4+0]; g = src[i*4+1]; b = src[i*4+2]; a = src[i*4+3]; }
        else if (channels == 3) { r = src[i*3+0]; g = src[i*3+1]; b = src[i*3+2]; a = 1.0f; }
        else { r = g = b = src[i]; a = 1.0f; }

        if (r < 0) r = 0; if (r > 1) r = 1;
        if (g < 0) g = 0; if (g > 1) g = 1;
        if (b < 0) b = 0; if (b > 1) b = 1;
        if (a < 0) a = 0; if (a > 1) a = 1;

        dst[i*4+0] = (u8)(r * 255.0f); dst[i*4+1] = (u8)(g * 255.0f);
        dst[i*4+2] = (u8)(b * 255.0f); dst[i*4+3] = (u8)(a * 255.0f);
    }
}

static bool _sdl_process_events(bool* running, int* win_w, int* win_h, SDL_Window* window, SDL_Renderer* renderer, SDL_Texture** texture, void** frame_buffer) {
    SDL_Event event;
    bool resized = false;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) *running = false;
        else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
            *win_w = event.window.data1;
            *win_h = event.window.data2;
            SDL_DestroyTexture(*texture);
            *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, *win_w, *win_h);
            free(*frame_buffer);
            *frame_buffer = malloc((size_t)*win_w * *win_h * 4);
            resized = true;
        }
    }
    return resized;
}

int sf_host_run(const sf_host_desc* desc) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) return 1;

    u32 flags = SDL_WINDOW_SHOWN;
    if (desc->resizable) flags |= SDL_WINDOW_RESIZABLE;
    if (desc->fullscreen) flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;

    SDL_Window* window = SDL_CreateWindow(desc->window_title ? desc->window_title : "SionFlow App",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, desc->width, desc->height, flags);
    if (!window) { SDL_Quit(); return 1; }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, desc->vsync ? SDL_RENDERER_PRESENTVSYNC : 0);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, desc->width, desc->height);

    sf_host_app app;
    if (sf_host_app_init(&app, desc) != 0) { SDL_DestroyWindow(window); SDL_Quit(); return 1; }

    u32* frame_buffer = malloc((size_t)desc->width * desc->height * 4);
    bool running = true;
    u32 start_ticks = SDL_GetTicks();
    f32 last_log_time = -desc->log_interval - 1.0f; 
    int win_w = desc->width, win_h = desc->height;

    while (running) {
        u32 current_ticks = SDL_GetTicks() - start_ticks;
        f32 current_time = current_ticks / 1000.0f;
        
        bool do_log = (desc->log_interval > 0) && (current_time - last_log_time) >= desc->log_interval;
        sf_log_set_global_level(do_log ? SF_LOG_LEVEL_TRACE : SF_LOG_LEVEL_WARN);
        if (do_log) { last_log_time = current_time; SF_LOG_INFO("--- Frame Log @ %.2fs ---", current_time); }

        _sdl_process_events(&running, &win_w, &win_h, window, renderer, &texture, (void**)&frame_buffer);
        
        int mx, my;
        u32 buttons = SDL_GetMouseState(&mx, &my);
        sf_host_inputs inputs = {
            .time = current_time, .width = win_w, .height = win_h,
            .mouse_x = (f32)mx, .mouse_y = (f32)my,
            .mouse_lmb = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0,
            .mouse_rmb = (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0
        };
        sf_host_app_update_inputs(&app, &inputs);

        sf_engine_error err = sf_host_app_step(&app);
        if (err != SF_ENGINE_ERR_NONE) {
            SF_LOG_ERROR("Engine failure: %s", sf_engine_error_to_str(err));
            running = false;
        }
        
        if (app.resources.output && frame_buffer) {
            convert_to_pixels(app.resources.output, frame_buffer, win_w * 4, win_w, win_h);
            SDL_UpdateTexture(texture, NULL, frame_buffer, win_w * 4);
        }
        
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        if (do_log && frame_buffer) {
            char shot_path[256]; time_t now = time(NULL); struct tm* t_struct = localtime(&now);
            strftime(shot_path, sizeof(shot_path), "logs/screenshot_%Y-%m-%d_%H-%M-%S.bmp", t_struct);
            SDL_Surface* ss = SDL_CreateRGBSurfaceFrom(frame_buffer, win_w, win_h, 32, win_w * 4, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
            if (ss) { SDL_SaveBMP(ss, shot_path); SDL_FreeSurface(ss); }
        }
    }
    
    free(frame_buffer);
    sf_host_app_cleanup(&app);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
