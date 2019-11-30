# Cheat Sheet

## Upload to Github

1. git add FILENAMES
2. git commit -m 'comments'
3. git push origin master

## Clear Cache: Clear files added by "git add" command
git ls-files 

git rm --cached FILENAME

## link to new local repo:
on local: 

generating ssh key:  ssh-keygen -t rsa

copy generated ssh key from ~/.ssh/id_rsa.pub

add new ssh key to github webpage in repo settings

in local: git clone (ssh address)

in local: git pull 
