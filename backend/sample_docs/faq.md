# Frequently Asked Questions

**Q: How do I request a new AWS permission?**
A: File a ticket in the #infra-requests Slack channel with the specific IAM policy
needed and your justification. Infra reviews requests daily.

**Q: Who owns the billing service?**
A: The Payments team owns `billing-service`. Reach them in #payments-team for any
questions about its APIs or data model.

**Q: What's the on-call rotation schedule?**
A: On-call rotates weekly, Monday to Monday, and is managed in PagerDuty under the
"Platform Primary" schedule. Swap requests should go through PagerDuty directly.

**Q: Where are architecture decision records (ADRs) kept?**
A: ADRs live in the `docs/adr` folder of the `platform` repo, one markdown file per
decision, numbered sequentially.
