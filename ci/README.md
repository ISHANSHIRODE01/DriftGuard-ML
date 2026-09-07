# CI workflow

`github-actions-ci.yml` must live at `.github/workflows/ci.yml` to run.

It could not be pushed via API (the token used lacked `workflow` scope).
To activate: on GitHub, use "Add file" -> "Create new file", name it
`.github/workflows/ci.yml`, and paste the contents of
`ci/github-actions-ci.yml`. The badge in README.md will go green on the
next push.
