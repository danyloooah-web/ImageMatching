# PowerShell script to download HPatches dataset
# Run this in PowerShell

$url = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"
$output = "datasets\hpatches-sequences-release.tar.gz"

Write-Host "Downloading HPatches dataset (1.4 GB)..."
Write-Host "This may take 5-15 minutes depending on your connection"
Write-Host ""

# Create datasets directory if it doesn't exist
if (!(Test-Path "datasets")) {
    New-Item -ItemType Directory -Path "datasets"
}

# Download with progress
try {
    Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
    Write-Host "Download complete!"
    Write-Host "File saved to: $output"
    Write-Host ""
    Write-Host "Now extract the file using 7-Zip or WinRAR"
    Write-Host "Or run: tar -xzf $output -C datasets"
} catch {
    Write-Host "Download failed. Error: $_"
    Write-Host ""
    Write-Host "Alternative: Download directly from browser:"
    Write-Host $url
}

