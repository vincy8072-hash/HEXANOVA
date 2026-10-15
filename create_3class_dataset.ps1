
$src = ".\drishti_dataset"
$dst = ".\drishti_3class_dataset"

if (-not (Test-Path "$src\train\images") -or
    -not (Test-Path "$src\train\labels")) {
    throw "Source dataset folders not found. Nothing copied."
}

if (Test-Path $dst) {
    throw "Destination already exists. Nothing overwritten."
}

$map = @{ "1" = "0"; "2" = "1"; "3" = "2" }

foreach ($split in @("train", "val", "test")) {
    $imgDir = Join-Path $src "$split\images"
    $lblDir = Join-Path $src "$split\labels"
    $outImg = Join-Path $dst "$split\images"
    $outLbl = Join-Path $dst "$split\labels"

    if (-not (Test-Path $imgDir) -or -not (Test-Path $lblDir)) {
        throw "Missing folders for $split."
    }

    New-Item -ItemType Directory -Force -Path $outImg, $outLbl | Out-Null

    $labelMap = @{}
    Get-ChildItem $lblDir -Filter "*.txt" -File | ForEach-Object {
        $labelMap[$_.BaseName] = $_
    }

    $copied = 0
    $excluded = 0
    $background = 0

    foreach ($img in Get-ChildItem $imgDir -File) {
        $label = $labelMap[$img.BaseName]

        if (-not $label) {
            Write-Warning "No label file: $($img.Name)"
            continue
        }

        $lines = @(Get-Content $label.FullName |
            Where-Object { $_.Trim() -ne "" })

        $ids = @($lines | ForEach-Object {
            ($_.Trim() -split '\s+')[0]
        })

        if (@($ids | Where-Object { $_ -notin @("1","2","3") }).Count -gt 0) {
            $excluded++
            continue
        }

        $newLines = @()
        foreach ($line in $lines) {
            $parts = $line.Trim() -split '\s+'
            if ($parts.Count -ne 5) {
                throw "Malformed label: $($label.FullName)"
            }
            $newLines += "$($map[$parts[0]]) $($parts[1]) $($parts[2]) $($parts[3]) $($parts[4])"
        }

        Copy-Item $img.FullName $outImg
        Set-Content (Join-Path $outLbl $label.Name) -Value $newLines

        if ($lines.Count -eq 0) { $background++ }
        $copied++
    }

    Write-Host "$split : Copied=$copied Excluded=$excluded Background=$background"
}

$yaml = @"
path: C:/Users/vincy/OneDrive/Desktop/HEXANOVA-main/drishti_3class_dataset
train: train/images
val: val/images
test: test/images
task: detect
nc: 3
names:
  0: submarine_pipeline
  1: shipwreck
  2: ghost_net
"@

Set-Content "$dst\drishti_3class.yaml" $yaml
Write-Host "Dataset creation finished. Original dataset unchanged."