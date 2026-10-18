KEY=`python -c "import secrets; print(secrets.token_urlsafe(32))"`
cp -v server.cfg server.cfg.old
sed "s/^SECRET-KEY=.*/SECRET-KEY=$KEY/" server.cfg.old > server.cfg
