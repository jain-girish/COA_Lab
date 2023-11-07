#ifndef __STALLS_H__
#define __STALLS_H__

enum counters {
    IDLE,
    SCOREBOARD,
    PIPELINE
};
extern unsigned long long STALLS[3];

#endif