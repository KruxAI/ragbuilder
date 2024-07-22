echo Installing RagBuilder
setlocal EnableDelayedExpansion

echo Downloading Brewfile...
curl -fsSL https://raw.githubusercontent.com/KruxAI/ragbuilder/main/Brewfile -o Brewfile

echo Reading Brewfile...
echo Installing packages from Brewfile...
for /f "usebackq tokens=*" %%i in ("Brewfile") do (
    python -m pip install %%i
)

echo Installing ragbuilder...
python -m pip install ragbuilder

echo Setup completed successfully.
endlocal
