# Runbook: Deploying a Service

## Overview
Deploys are triggered by merging to `main`. CI builds a container image, runs the
test suite, and pushes to the internal registry. A deploy bot then rolls the image
out to staging automatically.

## Promoting to production
Production promotion is manual. Open the `#deploys` Slack channel, post the staging
build URL, and get a thumbs-up reaction from an on-call engineer. Then run:

```
deploy promote --service <name> --env production
```

## Rollback
If a deploy causes errors, run `deploy rollback --service <name>` to revert to the
previous known-good image. Rollbacks take about 2 minutes to fully propagate.

## On-call escalation
If a rollback doesn't resolve the incident within 10 minutes, page the secondary
on-call via PagerDuty and open an incident channel using `/incident start`.
