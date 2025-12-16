#ifndef _API_TEST_HANDLERS_H_
#define _API_TEST_HANDLERS_H_

#include "cli_parser.h"

// Frame search API test handler functions
int handleTestFrameCache(ProgramArgs* cli);
int handleTestProgressApi(ProgramArgs* cli);
int handleTestSummaryApi(ProgramArgs* cli);

// Strip search API test handler functions
int handleTestStripProgressApi(ProgramArgs* cli);
int handleTestStripSummaryApi(ProgramArgs* cli);
int handleTestStripCache(ProgramArgs* cli);
int handleTestStripCompletionApi(ProgramArgs* cli);

#endif