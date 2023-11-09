#ifndef __STALLS_H__
#define __STALLS_H__

enum counters {
    IDLE,
    SCOREBOARD,
    PIPELINE
};
extern unsigned STALLS[28][3];

#endif