@echo off
REM Set nnU-Net environment variables for Windows

set nnUNet_raw=%~dp0nnUNet_raw
set nnUNet_preprocessed=%~dp0nnUNet_preprocessed
set nnUNet_results=%~dp0nnUNet_results

echo nnU-Net environment variables set:
echo nnUNet_raw=%nnUNet_raw%
echo nnUNet_preprocessed=%nnUNet_preprocessed%
echo nnUNet_results=%nnUNet_results%
echo.
echo Environment ready for nnU-Net!
