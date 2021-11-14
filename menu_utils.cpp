#include "menu_utils.h"

String awaitString(String iname) {
    char msg[128];
    if (iname.length()) {
        sprintf(msg, "Enter the %s:", iname.c_str());
        Serial.println(msg);
    }

    // Wait for data
    while(!Serial.available()) {}

    return Serial.readStringUntil("\n");
}

int awaitInteger(String iname, int minval, int maxval) {
    char msg[128];
    if (iname.length()) {
        sprintf(msg, "Enter the %s (%d-%d):", iname.c_str(), minval, maxval);
        Serial.println(msg);
    }

    // Wait for data
    while(!Serial.available()) {}

    // Parse it as an integer and discard the linefeed
    int i = Serial.parseInt();
    Serial.read();

    // Check the value
    if (minval > 0 && maxval > 0 && (i < minval || i > maxval)) {
        sprintf(msg, "ERROR: %s must be %d to %d! You specified %d.", iname, minval, maxval, i);
        Serial.println(msg);
        return -1;
    }
    else {
        return i;
    }
}
