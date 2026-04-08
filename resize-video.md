(base) spider@spiderscstudio2 screenlens % ffmpeg -i input-videos/ODC/pascalbornetNewsletterODC1.2.mov -filter:v "setpts=0.5*PTS" -an input-videos/ODC/pascal-half-no-audio.mov 
ffmpeg version 8.1 Copyright (c) 2000-2026 the FFmpeg developers
  built with Apple clang version 17.0.0 (clang-1700.6.4.2)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/8.1 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gpl --enable-libsvtav1 --enable-libopus --enable-libx264 --enable-libmp3lame --enable-libdav1d --enable-libvpx --enable-libx265 --enable-openssl --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      60. 26.100 / 60. 26.100
  libavcodec     62. 28.100 / 62. 28.100
  libavformat    62. 12.100 / 62. 12.100
  libavdevice    62.  3.100 / 62.  3.100
  libavfilter    11. 14.100 / 11. 14.100
  libswscale      9.  5.100 /  9.  5.100
  libswresample   6.  3.100 /  6.  3.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input-videos/ODC/pascalbornetNewsletterODC1.2.mov':
  Metadata:
    major_brand     : qt  
    minor_version   : 0
    compatible_brands: qt  
    creation_time   : 2026-04-07T13:04:51.000000Z
    com.apple.quicktime.author: ReplayKitRecording
    com.apple.quicktime.full-frame-rate-playback-intent: 1
  Duration: 01:12:56.81, start: 0.000000, bitrate: 2473 kb/s
  Stream #0:0[0x1](und): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 2530x2720, 2416 kb/s, 19.01 fps, 60 tbr, 600 tbn (default)
    Metadata:
      creation_time   : 2026-04-07T13:04:51.000000Z
      handler_name    : Core Media Video
      vendor_id       : [0][0][0][0]
      encoder         : H.264
  Stream #0:1[0x2](und): Audio: aac (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 52 kb/s, start 0.060771 (default)
    Metadata:
      creation_time   : 2026-04-07T13:04:51.000000Z
      handler_name    : Core Media Audio
      vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (h264 (native) -> h264 (libx264))
Press [q] to stop, [?] for help
[libx264 @ 0xc30c29180] using cpu capabilities: ARMv8 NEON DotProd I8MM
[libx264 @ 0xc30c29180] profile High, level 5.2, 4:2:0, 8-bit
[libx264 @ 0xc30c29180] 264 - core 165 r3222 b35605a - H.264/MPEG-4 AVC codec - Copyleft 2003-2025 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=48 lookahead_threads=8 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mov, to 'input-videos/ODC/pascal-half-no-audio.mov':
  Metadata:
    major_brand     : qt  
    minor_version   : 0
    compatible_brands: qt  
    com.apple.quicktime.full-frame-rate-playback-intent: 1
    com.apple.quicktime.author: ReplayKitRecording
    encoder         : Lavf62.12.100
  Stream #0:0(und): Video: h264 (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 2530x2720, q=2-31, 60 fps, 15360 tbn (default)
    Metadata:
      encoder         : Lavc62.28.100 libx264
      creation_time   : 2026-04-07T13:04:51.000000Z
      handler_name    : Core Media Video
      vendor_id       : [0][0][0][0]
    Side data:
      CPB properties: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
frame=   80 fps=0.0 q=24.0 size=       0KiB time=00:00:07.80 bitrate=   0.0kbits/s speed=15.6x elaframe=  259 fps=257 q=31.0 size=     512KiB time=00:00:25.21 bitrate= 166.3kbits/s dup=0 drop=42 sframe=  454 fps=300 q=26.0 size=     768KiB time=00:00:33.51 bitrate= 187.7kbits/s dup=0 drop=80 sframe=  675 fps=335 q=29.0 size=     768KiB time=00:00:41.31 bitrate= 152.3kbits/s dup=0 drop=155 frame=  880 fps=349 q=31.0 size=    1024KiB time=00:00:45.28 bitrate= 185.3kbits/s dup=0 drop=293 frame= 1083 fps=358 q=31.0 size=    1280KiB time=00:00:48.66 bitrate= 215.5kbits/s dup=0 drop=441 frame= 1277 fps=362 q=29.0 size=    1536KiB time=00:00:52.50 bitrate= 239.7kbits/s dup=0 drop=576 frame= 1483 fps=367 q=31.0 size=    1536KiB time=00:00:56.48 bitrate= 222.8kbits/s dup=0 drop=693 frame= 1692 fps=373 q=31.0 size=    1792KiB time=00:01:00.70 bitrate= 241.9kbits/s dup=0 drop=818 frame= 1895 fps=375 q=29.0 size=    1792KiB time=00:01:04.86 bitrate= 226.3kbits/s dup=0 drop=949 frame= 2093 fps=377 q=31.0 size=    2048KiB time=00: