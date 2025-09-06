## Installation

1. Create the Apps Script:
  - Go to https://script.google.com
  - Create a new project
  - Replace the default code with the script above
  - Save the project
2. Deploy as Web App:
  - Click "Deploy" → "New deployment"
  - Choose type: "Web app"
  - Set execute as: "Me"
  - Set access: "Anyone" (or "Anyone with Google account" for more security)
  - Click "Deploy" and copy the Web App URL

## Calling the API (with curl)

Google doesn't work nicely with POST... So all APIs are GET.

Send Progress (GET):
curl -L "https://script.google.com/macros/s/AKfycbxU11EoXYKwgckXVfONFJdvM_QOVvTKwxzTzbwwwTUsqWMDpOx67yOfH5RsKhfTekpiow/exec?action=sendProgress&spreadsheetId=HIDDEN_SHEET_ID&frameComplete=false&frameIdx=12345&kernelIdx=2&chunkIdx=10&patternsPerSecond=500000&bestGenerations=250&bestPattern=123456789ABCDEF0&bestPatternBin=0001001100110011001100110011001100110011001100110011001100110011&test=false"

Get Best Result (GET) - unchanged:
curl -L "https://script.google.com/macros/s/AKfycbxU11EoXYKwgckXVfONFJdvM_QOVvTKwxzTzbwwwTUsqWMDpOx67yOfH5RsKhfTekpiow/exec?action=getBestResult&spreadsheetId=HIDDEN_SHEET_ID"
