
## Publishing blog posts to WordPress


### Example usage

1. Convert notebook to HTML using
   ```
   jupyter-nbconvert --to HTML --template basic --embed-images  blogposts/<postname>.ipynb
   ```
   This command will produce the HTML file `blogposts/<postname>.html`
2. Copy-paste the HTML from `blogposts/<postname>.html` into a new post (source).
3. Remove the first few lines (H1 heading) to avoid double title
4. Publish the post


### WordPress setup

1. I copied over default styles to my WordPress template.
2. I modified some styles settings for code blocks (to avoid margins).
3. I added styles to make figures slightly larger.


