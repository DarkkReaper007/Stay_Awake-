# GitHub Setup Script
# Connects local repository to GitHub and pushes code

Write-Host "Connecting to GitHub..." -ForegroundColor Green

# Remove existing remote if it exists
git remote remove origin 2>$null

# Add GitHub remote
git remote add origin https://github.com/DarkkReaper007/Stay_Awake-.git

# Rename branch to main (if needed)
git branch -M main

# Pull remote changes and merge (allow unrelated histories)
Write-Host "Pulling remote changes..." -ForegroundColor Yellow
git pull origin main --allow-unrelated-histories --no-edit

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "`nDone! Your project is now on GitHub!" -ForegroundColor Green
Write-Host "View at: https://github.com/DarkkReaper007/Stay_Awake-" -ForegroundColor Cyan
