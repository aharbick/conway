#ifndef _menu_utils_H
#define _menu_utils_H

#include <Arduino.h>

// Show a message requesting an integer for "iname" and check that it falls between minval and maxval (if both are > 0)
int awaitInteger(String iname, int minval, int maxval);

// Show a message requesting a string for "iname"
String awaitString(String iname);

#endif
