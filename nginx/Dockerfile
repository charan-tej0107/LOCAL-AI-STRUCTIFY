# ── Nginx for Local AI Structify ───────────────────────────────────────
# Minimal nginx image serving as a reverse proxy in front of Streamlit.
# The real configuration is mounted as a volume at runtime.
# ────────────────────────────────────────────────────────────────────────

FROM nginx:alpine

# Remove the default nginx config (replaced by docker-compose volume mount)
RUN rm /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
