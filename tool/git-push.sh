#echo "add..."
#git add .
echo "commit..."
git commit -m "$1"
echo "push..."
git push -u origin master
