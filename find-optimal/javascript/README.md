# Progress API

The code in progress-api.js provides an API to our C++ code for storing progress data through a simple REST API implemented in AppScript.

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
curl -L "https://script.google.com/macros/s/AKfycbxU11EoXYKwgckXVfONFJdvM_QOVvTKwxzTzbwwwTUsqWMDpOx67yOfH5RsKhfTekpiow/exec?action=sendProgress&apiKey=SECRET_KEY_VALUE&frameComplete=false&frameIdx=12345&kernelIdx=2&chunkIdx=10&patternsPerSecond=500000&bestGenerations=250&bestPattern=123456789ABCDEF0&bestPatternBin=0001001100110011001100110011001100110011001100110011001100110011&test=false"

Get Best Result (GET) - unchanged:
curl -L "https://script.google.com/macros/s/AKfycbxU11EoXYKwgckXVfONFJdvM_QOVvTKwxzTzbwwwTUsqWMDpOx67yOfH5RsKhfTekpiow/exec?action=getBestResult&apiKey=SECRET_KEY_VALUE"

# Utilities

The code in utils.js can be added to an app script similar to above and provide a variety of utilities usable in Google Sheets:

* bestPatternBinToDecimal(binaryString) - converts a string like 0011010110101001111101000010000000011101110000010011110111000110 to 3866890173849615814
* bestPatternBinToRLE(binaryString) - converts the string to RLE format that works in Golly, etc.

