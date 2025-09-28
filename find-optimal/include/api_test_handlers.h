#ifndef _API_TEST_HANDLERS_H_
#define _API_TEST_HANDLERS_H_

#include "cli_parser.h"

// API test handler functions
int handleTestFrameCache(ProgramArgs* cli);
int handleTestProgressApi(ProgramArgs* cli);
int handleTestSummaryApi(ProgramArgs* cli);

#endif