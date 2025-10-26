# 🌐 GitHub Pages Setup Instructions

This guide explains how to enable GitHub Pages for SUJBOT2 interactive documentation.

## 📁 Current Structure

```
docs/
├── index.html                    # Landing page with links to both pipelines
├── indexing_pipeline.html        # Phase 1-5 visualization
├── user_search_pipeline.html     # Phase 7 + 17 tools visualization
└── .nojekyll                     # Prevents Jekyll processing
```

## 🚀 Activation Steps

### 1. Push to GitHub

```bash
git add docs/
git add README.md
git commit -m "docs: Add interactive HTML documentation for GitHub Pages"
git push origin main
```

### 2. Enable GitHub Pages

1. Go to your GitHub repository: `https://github.com/ADS-teamA/Advanced`
2. Click **Settings** (top right)
3. Scroll down to **Pages** (left sidebar)
4. Under **Source**, select:
   - **Branch:** `main`
   - **Folder:** `/docs`
5. Click **Save**

### 3. Wait for Deployment

- GitHub will deploy your site (usually takes 1-2 minutes)
- You'll see a green checkmark when ready
- Your site will be live at: `https://ads-teama.github.io/Advanced/`

### 4. Verify URLs

Once deployed, test these URLs:

- **Landing page:** https://ads-teama.github.io/Advanced/
- **Indexing Pipeline:** https://ads-teama.github.io/Advanced/indexing_pipeline.html
- **User Search Pipeline:** https://ads-teama.github.io/Advanced/user_search_pipeline.html

## 📝 Notes

### .nojekyll File

The `.nojekyll` file in `/docs` tells GitHub Pages to skip Jekyll processing. This is important because:
- Jekyll would ignore files/folders starting with `_`
- We want raw HTML served as-is
- Faster deployment (no build step)

### Automatic Updates

Once enabled, **any push to `main` branch** that modifies files in `/docs` will automatically update the live site!

```bash
# Example: Update documentation
vim docs/indexing_pipeline.html
git add docs/indexing_pipeline.html
git commit -m "docs: Update Phase 3 description"
git push origin main

# GitHub Pages will auto-deploy in ~1 minute
```

### Custom Domain (Optional)

If you want a custom domain like `docs.sujbot2.com`:

1. Add a `CNAME` file to `/docs`:
   ```bash
   echo "docs.sujbot2.com" > docs/CNAME
   git add docs/CNAME
   git commit -m "docs: Add custom domain"
   git push
   ```

2. In GitHub Settings → Pages, enter your custom domain
3. Configure DNS at your domain provider:
   - Add CNAME record: `docs` → `ads-teama.github.io`

## 🎨 Features in Our Documentation

### Landing Page (`index.html`)
- Clean, modern design
- 2 large clickable cards for each pipeline
- Stats overview (7 phases, 17 tools, -67% failures, etc.)
- GitHub link
- Responsive design (mobile-friendly)

### Pipeline Pages
- Interactive hover effects
- Code examples with syntax highlighting
- Research-backed notes (yellow boxes)
- Data flow diagrams (blue boxes)
- Performance tags
- Detailed tool breakdowns

## 🔧 Troubleshooting

### Site not showing?
- Check GitHub Actions tab for deployment status
- Verify `/docs` folder exists in `main` branch
- Make sure `.nojekyll` file is present

### 404 errors?
- File paths are case-sensitive
- Check file names exactly match URLs
- Clear browser cache

### Styling broken?
- All CSS is inline in each HTML file
- No external dependencies
- Should work immediately

## 📊 Benefits

✅ **Free hosting** (GitHub Pages is free for public repos)
✅ **Auto-deploy** (push to main = instant update)
✅ **HTTPS included** (secure by default)
✅ **Fast loading** (GitHub CDN)
✅ **Version control** (HTML changes tracked in git)
✅ **No build step** (pure HTML, no compilation)

## 🎯 Next Steps

After enabling GitHub Pages:

1. ✅ Update README.md with live links (already done)
2. ✅ Share links with team
3. 📤 Consider adding to project documentation
4. 🔗 Link from other documentation sources

---

**Questions?** Check [GitHub Pages documentation](https://docs.github.com/en/pages)
