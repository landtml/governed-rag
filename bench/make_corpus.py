"""Deterministic synthetic knowledge base and golden set for the governed_rag benchmark.

uv run python bench/make_corpus.py [--out bench/corpus] [--golden bench/golden.jsonl] [--seed 0]

Only random.Random(seed) and random.Random(seed + 1) (near-miss picks) are used. Every date is derived from REFERENCE.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
REFERENCE = date(2026, 11, 10)
REVIEW_BY = "2027-09-01"
EXPIRED_REVIEW_BY = "2026-06-01"
OWNERS = {
    "public": "policy-desk@kestrel.example",
    "support": "support-ops@kestrel.example",
    "engineering": "platform-eng@kestrel.example",
    "finance": "finance-ops@kestrel.example",
}
PRODUCTS = ("Kestrel Sync", "Kestrel Vault", "Kestrel Relay", "Kestrel Atlas", "Kestrel Beacon",
            "Kestrel Ledger", "Kestrel Forge", "Kestrel Pulse", "Kestrel Harbor", "Kestrel Lens",
            "Kestrel Notary", "Kestrel Compass")
GROUPS = ("public", "support", "engineering", "finance", "hr")

# Curated topics that supersede an archive/legacy document. All are public.
SUPERSEDING = frozenset({
    "refund-filing-window", "data-deletion-grace", "subscription-downgrade-notice",
    "starter-storage-quota", "webhook-retry-limit", "password-history-depth", "sso-group-member-limit",
    "sales-success-handoff", "qualified-lead-window", "sandbox-data-retention",
    "renewal-reminder-lead", "invitation-link-expiry",
})
# Curated topics whose review_by is already past on the reference date. All are public.
EXPIRED = frozenset({
    "wiki-review-cycle", "guest-wifi-pass", "laptop-loan-period", "visitor-badge-validity",
    "shared-calendar-retention",
})


class Topic(NamedTuple):
    area: str
    slug: str
    title: str
    key: str
    unit: str
    fact: str        # one sentence, contains {v} once
    procedure: str   # non-numeric rule
    filler_a: str
    filler_b: str
    ask: str         # noun phrase for "What is the ... at Kestrel Cloud?"
    proc_ask: str    # gerund phrase for "What is the process for ...?"
    group: str


TOPICS = tuple(Topic(*row) for row in (
    # ---- policy (15)
    ("policy", "refund-filing-window", "Refund filing window", "refund_filing_days", "days",
     "Customers may file a refund request for up to {v} days after the original purchase date.",
     "Requests are filed through the billing portal by the account owner, who attaches the original invoice.",
     "The window is counted in calendar days from the invoice date shown on the receipt, not from delivery.",
     "Refunds approved inside the window are paid back to the original payment method within ten business days.",
     "refund filing window", "filing a refund request", "public"),
    ("policy", "trial-length", "Free trial length", "trial_days", "days",
     "New workspaces receive a free trial lasting {v} days from the first successful sign-in.",
     "Trial extensions are granted by the account executive in the CRM record, never by the support desk.",
     "A trial workspace can invite collaborators without charge, and each invitation is logged in the admin console.",
     "Trial data is kept on the same retention schedule as paid data once the workspace converts to a subscription.",
     "free trial length", "extending a free trial", "public"),
    ("policy", "cancellation-notice", "Cancellation notice period", "cancellation_notice_days", "days",
     "Subscribers who cancel a renewal must give at least {v} days of written notice before the renewal date.",
     "Notices are sent through the cancellation form and confirmed by the billing team within one business day.",
     "Notice received after the renewal date counts toward the following term instead of the current one.",
     "Partial-year cancellations are not refunded, so the notice date determines which invoice becomes final.",
     "cancellation notice period", "sending a cancellation notice", "public"),
    ("policy", "api-rate-limit", "Public API rate limit", "api_requests_per_minute", "requests per minute",
     "Each API key may send up to {v} requests per minute before the gateway begins refusing calls.",
     "Limit increases are requested by the customer's technical contact through the developer portal.",
     "Bursts are smoothed over a rolling one-minute window rather than a fixed clock minute.",
     "Limits apply per key, so rotating keys does not raise the total allowance for a workspace.",
     "public API rate limit", "raising the API request limit", "public"),
    ("policy", "seat-minimum", "Business plan seat minimum", "business_min_seats", "seats",
     "The Business plan requires a minimum of {v} paid seats on every renewal quote.",
     "Seat counts are reviewed by the account executive before each renewal quote is issued to the customer.",
     "Seats can be reduced at renewal but not in the middle of a billing term.",
     "Guest collaborators do not count toward the seat minimum, although every paid seat is billed in full.",
     "seat minimum on the Business plan", "changing the seat count on a renewal", "public"),
    ("policy", "contractor-account-expiry", "Contractor account expiry", "contractor_expiry_days", "days",
     "Contractor accounts expire automatically after {v} days unless the sponsoring manager renews them.",
     "Renewals are requested by the sponsor through the access portal, with the statement of work attached.",
     "Expired contractor accounts keep their data for thirty days before the records are removed.",
     "Security reviews every active contractor account at the start of each quarter and reports the list to the sponsors.",
     "contractor account expiry", "renewing a contractor account", "public"),
    ("policy", "audit-log-retention", "Audit log retention", "audit_log_days", "days",
     "Audit log entries for every workspace are kept online for {v} days before they move to cold storage.",
     "Exports of audit logs are requested through the security desk and recorded in the access register.",
     "Entries are immutable once written and cannot be edited by workspace administrators under any circumstances.",
     "A legal hold suspends the move to cold storage for a named workspace until counsel releases the hold.",
     "audit log retention period", "exporting audit logs for an external auditor", "public"),
    ("policy", "data-deletion-grace", "Account deletion grace period", "deletion_grace_days", "days",
     "After an account owner requests closure, the workspace stays recoverable for {v} days.",
     "Closure requests are confirmed by email to the owner before any data is marked for removal.",
     "During that time support can restore the workspace in full without a new contract being signed.",
     "When the period ends, the workspace, its attachments and its backups are removed from every region.",
     "account deletion grace period", "requesting closure of an account", "public"),
    ("policy", "enterprise-tier-minimum", "Enterprise tier minimum spend", "enterprise_min_usd", "USD",
     "The Enterprise tier starts at {v} USD of annual committed spend under a signed order form.",
     "Tier changes are approved by the commercial lead and recorded on the order form before provisioning.",
     "Enterprise customers receive a named technical account manager for the first ninety days after go-live.",
     "Customers below the threshold may still buy individual Enterprise features as paid add-ons.",
     "Enterprise tier minimum spend", "moving a customer up to the Enterprise tier", "public"),
    ("policy", "password-reset-validity", "Password reset link validity", "reset_link_minutes", "minutes",
     "Password reset links stay valid for {v} minutes after they are sent to the registered address.",
     "Reset requests are verified against the recovery email on file before any link is issued.",
     "A link that has already been used is rejected even if its time window has not yet run out.",
     "Workspace administrators can shorten the window for their own users from the security settings page.",
     "password reset link validity", "requesting a password reset", "public"),
    ("policy", "subscription-downgrade-notice", "Subscription downgrade notice", "downgrade_notice_days", "days",
     "A customer who wants to move to a lower plan must give {v} days of notice before the change takes effect.",
     "Downgrade requests are submitted in the billing portal by the account owner and acknowledged by email.",
     "Features outside the lower plan are switched off on the effective date, and the stored data remains in place.",
     "Downgrades requested mid-term take effect at the next renewal unless finance approves an earlier change.",
     "subscription downgrade notice period", "moving a subscription to a lower plan", "public"),
    ("policy", "starter-storage-quota", "Starter plan storage quota", "starter_storage_gb", "GB",
     "Each Starter plan workspace includes {v} GB of file storage shared across all of its members.",
     "Storage add-ons are sold through the billing portal and take effect within one hour of payment.",
     "Storage used by deleted files stays counted until an administrator empties the recycle bin.",
     "Workspaces that exceed the quota can still read their files but cannot upload new ones until they add storage.",
     "Starter plan storage quota", "buying extra storage for a Starter workspace", "public"),
    ("policy", "webhook-retry-limit", "Webhook delivery retry limit", "webhook_retries", "attempts",
     "A failed webhook delivery is retried up to {v} attempts before the event is parked for manual review.",
     "Customers can replay parked events from the developer console after fixing their receiving endpoint.",
     "Each attempt is spaced further apart than the one before, so a slow endpoint is not flooded with retries.",
     "Parked events are kept for thirty days and are never delivered again automatically once they expire.",
     "webhook delivery retry limit", "retrying a failed webhook delivery", "public"),
    ("policy", "password-history-depth", "Password history depth", "password_history_count", "passwords",
     "A new password may not match any of the last {v} passwords used on the same account.",
     "The history check runs in the identity service whenever a user changes a password in any client.",
     "Passwords set by an administrator during an account recovery count toward the history as well.",
     "Users who are locked out are asked to contact the security desk rather than work around the history rule.",
     "password history depth", "choosing a new password after a change request", "public"),
    ("policy", "sso-group-member-limit", "Single sign-on group member limit", "sso_group_members", "members",
     "A single sign-on group may hold no more than {v} members before it must be split.",
     "Group splits are requested by the workspace administrator and reviewed by the identity team.",
     "Members copied into a second group keep their original assignments until the split is complete.",
     "The limit applies to directory-synced groups as well as to groups created by hand in the console.",
     "single sign-on group member limit", "splitting a large single sign-on group", "public"),
    # ---- procedure (15)
    ("procedure", "postmortem-deadline", "Postmortem completion deadline", "postmortem_days", "days",
     "Every severity one incident needs a published postmortem within {v} days of resolution.",
     "The incident commander drafts the postmortem and assigns each follow-up task to a named owner.",
     "Blameless language is checked by a second reviewer before the document is shared outside the team.",
     "Published postmortems are filed in the reliability wiki under the incident number for future reference.",
     "postmortem completion deadline", "writing a postmortem after a major outage", "public"),
    ("procedure", "offboarding-access-removal", "Offboarding access removal deadline", "offboarding_hours", "hours",
     "Access for a departing employee must be removed within {v} hours of the last working day.",
     "The people team files the offboarding ticket and the line manager confirms that each system is closed.",
     "Shared passwords held by the leaver are rotated during the same change window as the account closure.",
     "Equipment return is tracked by the facilities desk until the asset record shows the device as received.",
     "offboarding access removal deadline", "offboarding a departing employee", "public"),
    ("procedure", "maintenance-notice-period", "Planned maintenance notice period", "maintenance_notice_days", "days",
     "Planned maintenance that affects customer logins is announced at least {v} days in advance.",
     "The service owner publishes the notice on the status page and emails each workspace administrator directly.",
     "Each notice states the start time in UTC and the expected duration of the interruption.",
     "Maintenance that runs past the announced window is reported as an incident and follows the incident process.",
     "maintenance notice period", "announcing planned downtime to customers", "public"),
    ("procedure", "quote-validity-period", "Price quote validity period", "quote_validity_days", "days",
     "Price quotes remain valid for {v} days after they are issued to the customer.",
     "Quotes are generated only from the approved price book and carry the quote number on every page.",
     "A quote that has expired must be reissued by the account executive rather than extended by email.",
     "Customers who sign after expiry receive the price in force on the signature date, not the quoted one.",
     "price quote validity period", "reissuing an expired price quote", "public"),
    ("procedure", "crisis-comms-approval", "Security incident statement approval window", "comms_approval_minutes", "minutes",
     "Public statements about a security incident need approval within {v} minutes of the request.",
     "The communications lead routes each draft to legal and to the incident commander at the same time.",
     "Approved wording is pasted into the status page template without any edits by the publishing engineer.",
     "Social media posts repeat the approved text exactly and never add new factual claims.",
     "security incident statement approval window", "drafting a public statement about a security incident", "public"),
    ("procedure", "sales-success-handoff", "Sales to success handoff deadline", "handoff_hours", "hours",
     "Accounts moving from sales to customer success are introduced within {v} hours of signature.",
     "The account executive records the handoff in the CRM with the kickoff date and the contact list.",
     "The first success review is booked automatically for thirty days after go-live by the scheduling tool.",
     "Commitments made during the sale are copied into the handoff note word for word so nothing is lost.",
     "sales to success handoff deadline", "handing an account from sales to customer success", "public"),
    ("procedure", "customer-data-export-size", "Self-service data export size limit", "export_limit_gb", "GB",
     "A self-service data export is capped at {v} GB per request in the admin console.",
     "Larger exports are scheduled by a support engineer and delivered through an encrypted storage bucket.",
     "Export files expire from the download page after seven days to limit exposure of customer records.",
     "Each export is signed and its checksum is shown beside the download link for verification.",
     "self-service data export size limit", "requesting a large customer data export", "support"),
    ("procedure", "ticket-reopen-window", "Ticket reopen window", "reopen_days", "days",
     "A customer may reopen a closed ticket without a new charge for {v} days after closure.",
     "Reopened tickets keep their original number and return to the queue of the agent who closed them.",
     "Agents add a short note explaining why the case was reopened so that the history stays readable.",
     "Tickets closed for longer than the window open as a new case with a fresh number and a new clock.",
     "ticket reopen window", "reopening a ticket that was closed", "support"),
    ("procedure", "status-page-update-interval", "Status page update interval", "status_interval_minutes", "minutes",
     "During a live incident the status page is refreshed at least every {v} minutes.",
     "The incident lead posts each update in the status tool and mirrors it to the support channel.",
     "Updates state what is known, what is being done and when the next update will arrive.",
     "The final update is posted only after an engineer has verified that the affected service is healthy.",
     "status page update interval", "posting an update during a live outage", "support"),
    ("procedure", "shared-vault-review-cycle", "Shared vault access review cycle", "vault_review_days", "days",
     "Access to each shared password vault is reviewed by its owner every {v} days.",
     "Reviewers remove any member who has not signed in to the vault during the review period.",
     "Removals are logged in the vault activity feed together with the name of the reviewer.",
     "Vault owners who miss a review lose edit rights until they complete it through the admin console.",
     "shared vault access review cycle", "reviewing members of a shared password vault", "support"),
    ("procedure", "vendor-security-review", "Vendor security review turnaround", "vendor_review_days", "days",
     "New software vendors must complete the security review within {v} days of their first request.",
     "Procurement sends the questionnaire to the vendor and forwards the answers to the security team.",
     "Vendors that store customer data are reviewed again every year without exception.",
     "Exceptions are approved only by the chief information security officer, in writing, before any contract is signed.",
     "vendor security review turnaround", "onboarding a new software vendor", "finance"),
    ("procedure", "invoice-dispute-window", "Invoice dispute window", "dispute_window_days", "days",
     "A customer may dispute a charge on an invoice for {v} days after the invoice is issued.",
     "Disputes are opened by the account owner in the billing portal with the invoice number attached.",
     "Disputed amounts are held in a suspense ledger until the finance team reaches a decision.",
     "Support does not reverse charges directly, so every dispute passes through the finance queue.",
     "invoice dispute window", "disputing a charge on an invoice", "finance"),
    ("procedure", "on-call-handover-lead", "On-call handover lead time", "handover_minutes", "minutes",
     "The outgoing on-call engineer posts the handover note at least {v} minutes before the shift ends.",
     "The handover note is written in the on-call channel and linked from the pager schedule entry.",
     "The incoming engineer acknowledges the note in the channel before taking the pager over.",
     "Open incidents are transferred by name so that nothing is handed over implicitly between engineers.",
     "on-call handover lead time", "handing over the pager at shift change", "engineering"),
    ("procedure", "release-freeze-length", "Release freeze before quarter close", "freeze_hours", "hours",
     "The production release freeze before a quarter close lasts {v} hours in total.",
     "The release manager announces the freeze in the engineering channel at least one week ahead.",
     "Emergency fixes during the freeze need sign-off from the vice president of engineering.",
     "Hotfix branches are merged only after the change advisory board has reviewed the diff.",
     "release freeze length before quarter close", "requesting an exception to a release freeze", "engineering"),
    ("procedure", "change-board-lead-time", "Change board lead time", "change_lead_days", "days",
     "Production changes must be submitted to the change board at least {v} days before their window.",
     "Requests are filed in the change tracker with a rollback plan attached by the author of the change.",
     "The board meets on Tuesdays, and late requests wait for the following meeting.",
     "Emergency changes skip the queue but are reviewed by the board within one business day afterwards.",
     "change board lead time", "filing a production change request", "engineering"),
    # ---- definition (15)
    ("definition", "active-user-window", "Active user inactivity window", "active_window_days", "days",
     "An active user is anyone who has signed in during the last {v} days.",
     "Service accounts and bots are never counted as active users, even when they make API calls.",
     "Seat reports refresh every night at midnight in the workspace time zone set by the administrator.",
     "Deactivated users keep their history but no longer consume a paid seat in the billing system.",
     "active user inactivity window", "counting service accounts in seat reports", "public"),
    ("definition", "qualified-lead-window", "Lead qualification window", "qualify_days", "days",
     "A lead is qualified when it has replied within {v} days of the first contact.",
     "Sales development representatives tag the lead in the CRM once a reply has been logged.",
     "Leads that go silent are recycled into the nurture sequence by the marketing team.",
     "A lead can be qualified by only one representative, which prevents double counting in the pipeline.",
     "lead qualification window", "tagging a lead as qualified", "public"),
    ("definition", "severity-one-threshold", "Severity one workspace threshold", "sev1_workspaces", "workspaces",
     "A severity one incident is any outage affecting more than {v} customer workspaces at the same time.",
     "Any engineer or support agent may declare a severity one incident in the paging tool.",
     "Declaring one starts the incident bridge, the status page update and the executive page together.",
     "Only the incident commander may downgrade a declared incident, and only after a documented review.",
     "severity one workspace threshold", "declaring a severity one incident", "support"),
    ("definition", "dormant-workspace-threshold", "Dormant workspace threshold", "dormant_days", "days",
     "A workspace becomes dormant once it has gone {v} days without a single sign-in.",
     "Dormant workspaces are flagged to the account owner by an automatic email from the billing system.",
     "Billing continues for dormant workspaces until the owner cancels them or moves them to a lower plan.",
     "Data in a dormant workspace is never deleted without a separate, written closure request from the owner.",
     "dormant workspace threshold", "handling a workspace that has gone quiet", "public"),
    ("definition", "recovery-point-objective", "Recovery point objective", "rpo_minutes", "minutes",
     "The recovery point objective for customer databases allows at most {v} minutes of data loss.",
     "Restore drills run each quarter and the result is recorded in the reliability register.",
     "Replication lag above the objective pages the storage on-call engineer automatically.",
     "The objective applies to the primary region and to the standby region alike, with no exceptions.",
     "recovery point objective for customer databases", "running a restore drill", "engineering"),
    ("definition", "service-account-password-rotation", "Service account password rotation", "rotation_days", "days",
     "Service account passwords must be rotated at least every {v} days by the owning team.",
     "Rotation is performed through the secrets manager, which updates every consumer that reads the credential.",
     "A rotation that breaks a consumer is rolled back within the same change window by its owner.",
     "Service accounts without a named owner are disabled during the monthly inventory review.",
     "service account password rotation", "rotating a shared service credential", "engineering"),
    ("definition", "customer-key-rotation", "Customer-managed key rotation", "cmk_rotation_days", "days",
     "Customer-managed encryption keys are rotated automatically every {v} days unless the customer opts out.",
     "Customers who opt out must rotate keys themselves through the key management console and confirm the change in writing.",
     "Old key versions remain available for decryption so that existing backups stay readable.",
     "Rotation events are listed in the audit log and notified to the workspace security contact.",
     "customer-managed key rotation interval", "rotating an encryption key that the customer controls", "public"),
    ("definition", "workspace-owner-transfer", "Workspace owner transfer window", "transfer_hours", "hours",
     "A workspace ownership transfer is confirmed by both parties within {v} hours of the request.",
     "The outgoing owner starts the transfer in the admin console and the incoming owner accepts it.",
     "Until the transfer is accepted, the outgoing owner keeps full administrative rights to the workspace.",
     "Transfers between two different customers are never completed without a signed letter from both accounts.",
     "workspace ownership transfer window", "handing over ownership of a workspace", "public"),
    ("definition", "guest-collaborator-limit", "Guest collaborator limit", "guest_limit", "guests",
     "A paid workspace may invite up to {v} guest collaborators who hold no paid seat.",
     "Guests are invited by a workspace administrator and are listed separately in the membership report.",
     "A guest sees only the boards and folders that were explicitly shared with that guest.",
     "Guest invitations expire automatically when the workspace is moved to a plan that does not allow guests.",
     "guest collaborator limit", "inviting outside collaborators without a seat", "public"),
    ("definition", "sla-first-response", "First response time for paid tiers", "first_response_minutes", "minutes",
     "The first response to a paid-tier ticket is due within {v} minutes of the ticket being opened.",
     "The response clock starts when the ticket arrives in the queue, not when the customer finishes typing.",
     "A response counts when an agent posts a reply in the ticket thread, and an automated acknowledgement does not.",
     "Clock pauses are recorded by the agent when the ticket is waiting on the customer for more information.",
     "first response time for paid tiers", "measuring how quickly paid customers hear back", "support"),
    ("definition", "sandbox-data-retention", "Sandbox data retention", "sandbox_days", "days",
     "Data in a sandbox environment is deleted automatically after {v} days without any activity.",
     "Sandboxes can be kept longer by the account owner through a request in the developer portal.",
     "Test data copied from production is masked before it is placed in any sandbox workspace.",
     "Sandbox credentials are revoked on the same day the sandbox itself is deleted by the system.",
     "sandbox data retention period", "keeping an idle sandbox workspace", "public"),
    ("definition", "renewal-reminder-lead", "Renewal reminder lead time", "reminder_days", "days",
     "Account owners receive a renewal reminder email at least {v} days before the renewal date.",
     "Reminders are generated by the billing system and copied to the account executive for the account.",
     "A reminder is not sent for a subscription that is already marked for cancellation.",
     "Customers can change the reminder schedule in their billing settings without contacting support.",
     "renewal reminder lead time", "sending a renewal reminder to a customer", "public"),
    ("definition", "invitation-link-expiry", "Invitation link expiry", "invite_hours", "hours",
     "An invitation link sent to a new member stops working after {v} hours.",
     "A new invitation is issued by the administrator from the membership page, which shows the expiry time.",
     "An expired link can be reissued by the administrator at any time without a new order.",
     "Accepting an invitation after the link expires shows the new member a message asking for a fresh link.",
     "invitation link expiry", "reissuing an expired invitation link", "public"),
    ("definition", "leave-request-notice", "Leave request notice period", "leave_notice_days", "days",
     "Staff must submit a leave request at least {v} days before the first day of absence.",
     "Requests are made in the people system and approved by the line manager within two working days.",
     "Sick leave is exempt from the notice period but must be reported on the first day of absence.",
     "Leave that overlaps a release freeze needs the approval of the engineering director as well.",
     "leave request notice period", "asking for time off in advance", "public"),
    ("definition", "callback-quota", "Daily outbound callback quota", "callback_quota", "calls",
     "Each support agent may place up to {v} outbound callbacks during a single shift.",
     "Callbacks are logged in the telephony tool before the agent begins the next call in the queue.",
     "Quota overruns require the team lead to approve the extra calls in writing before they are placed.",
     "Callbacks to numbers on the do-not-call list are blocked automatically by the telephony system.",
     "daily outbound callback quota", "placing an outbound callback to a customer", "support"),
    # ---- internal (15)
    ("internal", "code-review-turnaround", "First code review turnaround", "review_hours", "hours",
     "A pull request should receive its first review within {v} hours of being opened.",
     "Reviewers are assigned automatically by the team channel bot, using the code owners file in the repository.",
     "A pull request with more than four hundred changed lines is split by its author before review starts.",
     "Draft pull requests are excluded from the turnaround clock until the author marks them ready for review.",
     "first code review turnaround", "getting a pull request reviewed quickly", "engineering"),
    ("internal", "expense-approval-limit", "Expense approval limit", "expense_limit_usd", "USD",
     "Single expenses above {v} USD need approval from a second manager before payment.",
     "Approvals are recorded in the expense tool with the vendor receipt attached to the claim.",
     "Split purchases made to stay under the limit are treated as a single expense for review.",
     "Travel bookings follow the same threshold unless the travel policy states otherwise for the trip.",
     "single expense approval limit", "approving a large expense claim", "finance"),
    ("internal", "purchase-order-threshold", "Purchase order threshold", "po_threshold_usd", "USD",
     "Any purchase above {v} USD requires a purchase order issued before the supplier is engaged.",
     "Requests are entered in the procurement tool by the budget holder and routed to finance.",
     "Verbal commitments to suppliers do not count as approval, even when the amount is small.",
     "Recurring subscriptions are reviewed annually and need a fresh purchase order at each renewal.",
     "purchase order threshold", "raising a purchase order for a supplier", "finance"),
    ("internal", "travel-booking-notice", "Travel booking advance notice", "travel_notice_days", "days",
     "Flights for company travel must be booked at least {v} days before departure.",
     "Bookings are made through the approved travel agency portal rather than on personal accounts.",
     "Late bookings above the fare cap are reviewed by the finance partner for the business unit.",
     "Rail journeys within the country are exempt from the advance booking requirement.",
     "travel booking advance notice", "booking a flight for a business trip", "finance"),
    ("internal", "supplier-payment-deadline", "Supplier payment deadline", "supplier_payment_days", "days",
     "Approved supplier invoices are paid within {v} days of receipt unless a contract says otherwise.",
     "Invoices are matched to purchase orders by the accounts payable clerk before they are released.",
     "Early payment discounts are considered only when the finance director signs the purchase order.",
     "Late invoices are escalated to the category owner in the weekly payables review meeting.",
     "supplier payment deadline", "paying a supplier invoice on time", "finance"),
    ("internal", "incident-bridge-idle", "Incident bridge idle timeout", "bridge_idle_minutes", "minutes",
     "The incident bridge call closes automatically after {v} minutes without a speaker.",
     "Before the call closes, the bot posts a reminder in the incident channel for the commander.",
     "Reopening the bridge takes one click from the incident page and keeps the same conference number.",
     "Recordings of the bridge are stored with the incident timeline, not on the personal drives of participants.",
     "incident bridge idle timeout", "keeping the incident call open during a quiet period", "support"),
    ("internal", "support-macro-review", "Support macro review interval", "macro_review_days", "days",
     "Every saved reply macro must be reviewed by its owner at least once every {v} days.",
     "Owners check the wording against the current policy and retire any macro that refers to an old feature.",
     "New macros are proposed in the support guild channel and approved by two senior agents before use.",
     "Macros that mention pricing are reviewed by finance in addition to the owner, without exception.",
     "support macro review interval", "keeping canned replies up to date", "support"),
    ("internal", "csat-survey-wait", "Satisfaction survey delay", "csat_delay_hours", "hours",
     "The satisfaction survey is sent no sooner than {v} hours after a ticket is closed.",
     "The delay gives the customer time to notice that the issue is resolved before being asked for a rating.",
     "Surveys are not sent to customers who have already declined two surveys in the same quarter.",
     "Survey results appear on the team dashboard but are never shown to the individual agent by name.",
     "satisfaction survey delay", "when customers get asked to rate a closed ticket", "support"),
    ("internal", "chat-transcript-retention", "Chat transcript retention", "chat_transcript_days", "days",
     "Live chat transcripts are stored for {v} days in the support archive.",
     "Agents tag each chat with a product area before closing the conversation in the chat console.",
     "Transcripts containing personal data are masked before they are shared outside the support team.",
     "Requests to delete a transcript early go to the privacy queue and are never handled by the support desk.",
     "chat transcript retention", "tagging a closed chat conversation", "support"),
    ("internal", "attachment-size-cap", "Maximum ticket attachment size", "attachment_max_mb", "MB",
     "Files attached to a support ticket may not exceed {v} MB each.",
     "Larger files are shared through a time-limited transfer link generated by the support agent.",
     "Compressed archives are measured at their compressed size when the limit is checked on upload.",
     "Customers are told the limit when the upload form opens so that they can split the file in advance.",
     "maximum ticket attachment size", "sending a file too large for a ticket", "support"),
    ("internal", "wiki-review-cycle", "Wiki page review cycle", "wiki_review_days", "days",
     "Every page in the internal wiki must be reviewed by its owner within {v} days of the last edit.",
     "Owners who do not review a page in time see it marked stale on the wiki index for their team.",
     "Stale pages stay readable but are moved to the archive section after a further ninety days.",
     "The wiki team runs a quarterly sweep to find pages that have no owner and assigns them to a lead.",
     "wiki page review cycle", "keeping internal wiki pages current", "public"),
    ("internal", "guest-wifi-pass", "Guest Wi-Fi pass validity", "guest_wifi_hours", "hours",
     "A guest Wi-Fi pass issued at reception stays valid for {v} hours from the time it is printed.",
     "Reception staff print passes from the visitor kiosk, which records the host name against each pass.",
     "Passes cannot be extended, so a visitor who needs longer access must receive a new pass from the host.",
     "The guest network is isolated from the corporate network and is never used for company devices.",
     "guest Wi-Fi pass validity", "giving a visitor internet access in the office", "public"),
    ("internal", "laptop-loan-period", "Loaner laptop loan period", "loan_days", "days",
     "A loaner laptop may be kept for up to {v} days while a repair is under way.",
     "Loans are requested through the IT desk, and the borrower signs the loan form at the depot.",
     "Loaners are imaged fresh before handover and wiped completely before they return to the shelf.",
     "Borrowers who keep a loaner past the period are charged the replacement value of the device.",
     "loaner laptop loan period", "borrowing a laptop during a repair", "public"),
    ("internal", "visitor-badge-validity", "Visitor badge validity", "badge_hours", "hours",
     "A visitor badge is valid for {v} hours and must be returned to reception on departure.",
     "Visitors sign in at the front desk and are escorted by their host at all times inside the building.",
     "Badges are printed with the visit date and the host name in large type on the front.",
     "Lost badges are reported to facilities at once so that the access card can be disabled.",
     "visitor badge validity", "how long a visitor can wear a badge", "public"),
    ("internal", "shared-calendar-retention", "Shared calendar retention", "calendar_days", "days",
     "Deleted events on shared team calendars are kept in the recovery folder for {v} days.",
     "Recovery requests are sent to the workplace IT desk, which restores events from the backup snapshot.",
     "Recurring meetings are restored as a single series so that attendee lists are preserved.",
     "Calendars of departed employees are archived to the mailbox of their line manager for a year.",
     "shared calendar retention", "restoring a deleted team calendar event", "public"),
))

CUSTOMERS = ("Harbor Dental", "Lantern Bakery", "Oakfield Surveying", "Brightline Pediatrics",
             "Corvin Logistics", "Maplestone Realty", "Tidewater Clinics", "Northpoint Academy",
             "Redfern Architects", "Sable Freight", "Pinecrest Credit Union", "Ironbridge Print",
             "Velma Foods", "Quarry Lane Books", "Summit Veterinary", "Juniper Staffing",
             "Keystone Pharmacy", "Ashgrove Council", "Bramble and Finch Studio", "Westmark Insurance")
PLANS = ("Starter", "Team", "Business", "Enterprise")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
OPENERS_A = ("{customer} on the {plan} plan wrote in on {day} because {symptom}.",
             "On {day}, {customer} from the {plan} plan contacted support because {symptom}.")
OPENERS_C = ("On {day}, {customer} on the {plan} plan said that {symptom}.",
             "{customer} ({plan} plan) told us on {day} that {symptom}.")

# (slug, product, symptom, cause, resolution, topic phrase for questions)
ISSUES = (
    ("sync-rename-loop", "Kestrel Sync", "files re-upload in a loop after a rename",
     "a stale folder index", "rebuilt the index from the admin console", "the endless re-upload loop after a rename"),
    ("sync-selective-hidden", "Kestrel Sync", "newly added folders disappear from selective sync",
     "a selection cache that was never refreshed", "cleared the selection cache on the device",
     "newly added folders vanishing from selective sync"),
    ("sync-conflict-copies", "Kestrel Sync", "the desktop client makes a conflict copy on every save",
     "clock skew on the workstation", "enabled network time sync and forced a resync",
     "conflict copies appearing on every save"),
    ("vault-unlock-after-password", "Kestrel Vault", "vault unlock fails right after a password change",
     "a cached credential token", "signed the user out of all sessions and cleared the token store",
     "vault unlock failing just after a password change"),
    ("vault-shared-readonly", "Kestrel Vault", "shared entries show as read-only for editors",
     "a role inherited from an old group membership", "removed the stale group membership and re-added the editor",
     "editors seeing shared entries as read-only"),
    ("vault-autofill-dead", "Kestrel Vault", "autofill stops working in a single browser profile",
     "an extension pinned to an old origin", "reinstalled the browser extension and reapproved the origin",
     "autofill going dead in one browser profile"),
    ("relay-duplicate-webhooks", "Kestrel Relay", "webhooks arrive twice with the same delivery id",
     "a retry after a timed-out acknowledgement", "raised the acknowledgement timeout and enabled delivery id deduplication",
     "duplicate webhook deliveries with identical ids"),
    ("relay-sandbox-queue", "Kestrel Relay", "outbound messages sit in the queue for hours in sandbox",
     "a throttled sandbox quota", "lifted the sandbox quota for the test account",
     "outbound messages stuck in the sandbox queue for hours"),
    ("relay-partner-tls", "Kestrel Relay", "TLS handshakes fail against one partner endpoint only",
     "an intermediate certificate missing from the chain", "uploaded the full certificate chain to the endpoint profile",
     "TLS handshakes failing against a single partner endpoint"),
    ("atlas-blank-dashboards", "Kestrel Atlas", "dashboards load blank after the date range is changed",
     "a cached query plan from an old schema", "purged the query plan cache for the workspace",
     "dashboards that go blank after a date range change"),
    ("atlas-monthly-report", "Kestrel Atlas", "scheduled monthly reports skip the last day of the month",
     "a schedule pinned to a UTC offset", "switched the schedule to the workspace named timezone",
     "scheduled monthly reports missing the final day"),
    ("atlas-csv-columns", "Kestrel Atlas", "CSV exports drop every column whose values contain commas",
     "a delimiter that was not escaped in the exporter", "switched the export to quoted fields",
     "CSV exports losing columns whose values contain commas"),
    ("beacon-retired-probe", "Kestrel Beacon", "status alerts fire for a service that is healthy",
     "a probe pointing at a retired hostname", "repointed the probe to the current load balancer",
     "status alerts firing for healthy services"),
    ("beacon-maintenance-gaps", "Kestrel Beacon", "uptime history shows gaps during planned maintenance",
     "a maintenance flag that was never applied to the probes", "applied the maintenance flag to the affected probes retroactively",
     "gaps in uptime history during planned maintenance"),
    ("beacon-late-sms", "Kestrel Beacon", "SMS alerts arrive hours late",
     "carrier throttling on a shared sender id", "moved alerts to a dedicated sender id",
     "SMS alerts arriving hours late"),
    ("ledger-rounding-cents", "Kestrel Ledger", "invoice totals differ by a few cents from the portal",
     "rounding applied per line instead of per invoice", "switched the rounding mode to invoice level",
     "invoice totals off by a few cents compared with the portal"),
    ("ledger-tax-certificate", "Kestrel Ledger", "tax exemption certificates are rejected as unreadable at upload",
     "a scan saved in the wrong orientation", "re-uploaded a corrected scan of the certificate",
     "tax exemption certificates rejected at upload"),
    ("ledger-open-month", "Kestrel Ledger", "ledger exports leave out the current open month",
     "an export window set to the prior close", "extended the export window to the open period",
     "ledger exports leaving out the open month"),
    ("forge-branch-artifact", "Kestrel Forge", "builds fail for a missing artifact after a branch rename",
     "a pipeline cache key tied to the branch name", "cleared the pipeline cache and re-keyed it on the commit hash",
     "builds failing for a missing artifact after a branch rename"),
    ("forge-double-deploy", "Kestrel Forge", "deploy hooks fire twice for the same tag",
     "duplicate trigger rules left behind by a migration", "removed the duplicate trigger rule",
     "deploy hooks firing twice for one tag"),
    ("forge-zero-runners", "Kestrel Forge", "the runner pool sits at zero capacity",
     "an autoscaler paused by a stale maintenance flag", "cleared the stale maintenance flag on the pool",
     "the runner pool sitting at zero capacity"),
    ("pulse-hour-lag", "Kestrel Pulse", "metric dashboards run about an hour behind real time",
     "an ingest pipeline falling behind on one partition", "rebalanced the partition assignment",
     "metric dashboards running an hour behind real time"),
    ("pulse-weekend-anomalies", "Kestrel Pulse", "anomaly detection flags every weekend",
     "a seasonality model trained on weekdays only", "retrained the seasonality model on a full calendar year",
     "anomaly detection flagging every weekend"),
    ("pulse-metric-prefix", "Kestrel Pulse", "custom metric names carry stray prefixes",
     "an SDK version mismatch in the agent", "upgraded the agent to the matching SDK release",
     "custom metric names carrying stray prefixes"),
    ("harbor-regional-pulls", "Kestrel Harbor", "registry pulls time out from one region",
     "a regional mirror that was out of date", "forced a mirror sync for that region",
     "registry pulls timing out in a single region"),
    ("harbor-fixed-cve", "Kestrel Harbor", "scans still list a vulnerability that was already fixed",
     "a scanner database older than the patch", "updated the vulnerability database and rescanned the images",
     "scans reporting a vulnerability that was already fixed"),
    ("harbor-retention-tags", "Kestrel Harbor", "the retention policy removes tagged release images",
     "a tag exemption rule missing a pattern", "added the release tag pattern to the exemption list",
     "retention policy removing tagged release images"),
    ("lens-blank-replay", "Kestrel Lens", "session replays are blank for mobile visitors",
     "a content security header blocking the recorder script", "adjusted the header to allow the recorder script",
     "blank session replays from mobile visitors"),
    ("lens-double-funnel", "Kestrel Lens", "funnel reports count checkout steps twice",
     "a duplicate tracking tag on the checkout page", "removed the second tracking tag from the template",
     "funnel reports counting checkout steps twice"),
    ("lens-frozen-heatmaps", "Kestrel Lens", "heatmaps have stopped updating since the site redesign",
     "page selectors recorded from the old layout", "rebuilt the page selectors from the new layout",
     "heatmaps frozen since the site redesign"),
    ("notary-expired-seal", "Kestrel Notary", "signed documents display an expired seal",
     "a certificate authority chain that was not refreshed", "reissued the seal with the renewed authority certificate",
     "signed documents showing an expired seal"),
    ("notary-envelope-link", "Kestrel Notary", "signers cannot open an envelope link they received",
     "a link generated before the sender changed domains", "regenerated the envelope link from the sender console",
     "signers unable to open an envelope link"),
    ("notary-witness-missing", "Kestrel Notary", "the audit trail omits a witness signature",
     "a witness added after the envelope was locked", "unlocked the envelope and re-recorded the witness step",
     "audit trails missing a witness signature"),
    ("compass-reordering", "Kestrel Compass", "roadmap items reorder themselves overnight",
     "a sort field stored as a locale string", "converted the sort field to a numeric rank",
     "roadmap items reordering themselves overnight"),
    ("compass-guest-board", "Kestrel Compass", "guests cannot see a shared board",
     "a guest role that lacks the board scope", "granted the board scope to the guest role",
     "guests locked out of a shared board"),
    ("compass-silent-notifications", "Kestrel Compass", "notifications stop after a workspace move",
     "a webhook secret tied to the old workspace", "regenerated the notification secret after the move",
     "notifications going silent after a workspace move"),
    ("sync-proxy-crawl", "Kestrel Sync", "uploads crawl to a near standstill on one office network",
     "a proxy inspecting large uploads", "added the sync service to the proxy bypass list",
     "uploads crawling on one office network"),
    ("vault-recovery-codes", "Kestrel Vault", "recovery codes are refused after a device reset",
     "codes generated for the previous device key", "regenerated the recovery codes on the new device",
     "recovery codes refused after a device reset"),
    ("relay-partition-order", "Kestrel Relay", "messages arrive out of order within one queue partition",
     "a consumer restarted without draining the partition", "drained the partition and restarted consumers in order",
     "messages arriving out of order within one queue partition"),
    ("atlas-missing-team-filter", "Kestrel Atlas", "row-level filters skip a team added last week",
     "a group sync delayed by a failed job", "re-ran the directory group sync job",
     "row-level filters skipping a newly added team"),
    ("atlas-monday-timeouts", "Kestrel Atlas", "dashboards time out every Monday morning",
     "a weekly rollup job overlapping the peak load", "moved the rollup job to the early morning slot",
     "dashboards timing out every Monday morning"),
    ("beacon-region-disagree", "Kestrel Beacon", "probe results disagree between two regions",
     "probe agents using different DNS resolvers", "pinned the agents to the internal resolver",
     "probe results disagreeing between two regions"),
    ("ledger-former-staff", "Kestrel Ledger", "payment reminders reach former employees",
     "a billing contact still set to a departed user", "reassigned the billing contact to the finance mailbox",
     "payment reminders reaching former employees"),
    ("ledger-credit-stuck", "Kestrel Ledger", "credit notes refuse to apply to new invoices",
     "a credit note already marked as consumed", "voided the consumed mark and reapplied the credit",
     "credit notes refusing to apply to new invoices"),
    ("forge-checks-silent", "Kestrel Forge", "pull request checks never start",
     "a webhook disabled after a repository transfer", "re-enabled the webhook from the repository settings",
     "pull request checks never starting"),
    ("forge-empty-artifact", "Kestrel Forge", "artifact downloads return an empty file",
     "an upload silently truncated part way through", "re-uploaded the artifact with checksum verification turned on",
     "artifact downloads returning empty files"),
    ("pulse-silence-early", "Kestrel Pulse", "alert silences expire before their set time",
     "silence durations stored in local time", "converted silence durations to UTC on save",
     "alert silences ending early"),
    ("pulse-recent-logs", "Kestrel Pulse", "log search returns nothing for recent entries",
     "an index rollover stuck on one shard", "forced the index rollover on the stuck shard",
     "log search returning nothing for recent entries"),
    ("harbor-robot-unauthorized", "Kestrel Harbor", "robot account pushes are rejected as unauthorized",
     "a robot secret rotated without updating the clients", "issued a new robot secret and updated the CI variables",
     "robot account pushes rejected as unauthorized"),
    ("harbor-wrong-arch", "Kestrel Harbor", "replicated images target the wrong CPU architecture",
     "a replication filter ignoring platform tags", "enabled platform matching in the replication rule",
     "replicated images targeting the wrong CPU architecture"),
    ("lens-consent-returns", "Kestrel Lens", "the consent banner reappears after visitors accept it",
     "a consent cookie set on a single subdomain only", "reissued the consent cookie for the parent domain",
     "the consent banner reappearing after visitors accept it"),
    ("notary-wrong-language", "Kestrel Notary", "reminder emails arrive in the wrong language",
     "a language taken from the sender profile", "switched reminders to the recipient language preference",
     "reminder emails arriving in the wrong language"),
    ("compass-leave-totals", "Kestrel Compass", "time tracking totals ignore approved leave",
     "leave records imported without the approved flag", "re-imported the leave records with the approval flag set",
     "time tracking totals ignoring approved leave"),
    ("compass-missing-fields", "Kestrel Compass", "custom fields are missing from exported reports",
     "a field deleted and recreated with the same name", "restored the field mapping to the new field identifier",
     "custom fields missing from exported reports"),
    ("sync-internal-login", "Kestrel Sync", "shared links send internal users to a login page",
     "a link policy set to external only", "changed the link policy to allow the internal domain",
     "shared links sending internal users to a login page"),
    ("vault-note-attachments", "Kestrel Vault", "secure notes refuse attachments after an import",
     "an import that created notes without a storage key", "ran the storage key repair tool on the imported notes",
     "secure notes refusing attachments after an import"),
    ("relay-weekend-credentials", "Kestrel Relay", "sandbox credentials fail on weekends",
     "sandbox keys on a fixed rolling expiry timer", "extended the sandbox key expiry for the account",
     "sandbox credentials failing at weekends"),
    ("atlas-stale-embeds", "Kestrel Atlas", "embedded charts show stale data",
     "an iframe cache that was not invalidated on refresh", "enabled cache invalidation on the embed token",
     "embedded charts showing stale data"),
    ("beacon-deploy-pages", "Kestrel Beacon", "heartbeat monitors page on-call during deployments",
     "a deploy pause missing from the monitor definition", "added a deploy pause window to the monitor",
     "heartbeat monitors paging on-call during deployments"),
    ("pulse-dark-tiles", "Kestrel Pulse", "dashboard tiles fail to render in dark mode",
     "a theme token missing from one widget template", "added the missing theme token to the widget template",
     "dashboard tiles failing to render in dark mode"),
)

HANDBOOK = (
    ("new-laptop-setup", "New laptop setup", (
        ("Imaging", "New laptops are imaged from the standard build at the IT depot before they are shipped to the new hire.",
         "where new laptops get their standard image"),
        ("Accounts", "The IT desk adds the password manager seat on the first morning, and only after the acceptable use form is signed.",
         "when a new hire's password manager seat is added"),
        ("Handover", "The line manager hands the device over in person and records the asset tag in ServiceDesk.",
         "who hands a new laptop to its user"))),
    ("first-week-buddy", "First week buddy", (
        ("Assignment", "The team lead picks a buddy from the same team before the start date and posts the name in the team channel.",
         "how a buddy is picked for a new hire"),
        ("Check-ins", "The buddy books a fifteen-minute check-in every morning of the first week in the team calendar.",
         "how often the buddy checks in during week one"),
        ("Summary", "On the Friday of the first week, the buddy sends a short written summary to the manager.",
         "what the buddy reports at the end of week one"))),
    ("code-review-etiquette", "Code review etiquette", (
        ("Comment labels", "Reviewers prefix optional comments with the word nit so that authors can skip them without discussion.",
         "how reviewers mark optional comments"),
        ("Re-review", "The reviewer who requested changes re-reviews the revised commit, rather than handing it to a different reviewer.",
         "who checks a revised commit after changes were requested"),
        ("Escalation", "A review thread that runs past three rounds moves to a short call in the engineering-pairing channel.",
         "where a review disagreement goes after several rounds"))),
    ("release-notes", "Release notes", (
        ("Drafting", "The release manager drafts the notes in Confluence on the day before the release train departs.",
         "where the release notes are drafted"),
        ("Audience", "Customer-facing notes are written in plain language and reviewed by the product marketing manager before publishing.",
         "who approves customer-facing release notes"),
        ("Archive", "Each published note is copied to the changelog folder with the release tag as its file name.",
         "how published release notes are archived"))),
    ("incident-comms-channel", "Incident communications channel", (
        ("Channel naming", "Every incident gets a dedicated Slack channel named inc followed by the incident number, created by the on-call engineer.",
         "how incident channels are named"),
        ("Quiet hours", "Nobody posts speculation in the incident channel, and questions go into the thread under the commander's update.",
         "where questions go during an active incident"),
        ("Closure", "The channel is archived by the commander once the postmortem link has been pinned at the top of it.",
         "when an incident channel gets archived"))),
    ("sprint-planning", "Sprint planning", (
        ("Capacity", "Planned capacity is set at sixty percent of available hours to leave room for support rotations and unplanned work.",
         "how much of each sprint is planned work"),
        ("Commitments", "Stories are committed in Jira only after the product owner has confirmed the acceptance criteria in writing.",
         "when a story is committed to a sprint"),
        ("Carryover", "Unfinished stories carry over to the next sprint only once, after which they return to the backlog for re-estimation.",
         "what happens to stories left unfinished in a sprint"))),
    ("retrospectives", "Retrospectives", (
        ("Format", "Retrospectives use the start, stop and continue format on a shared board created by the scrum master.",
         "what format retrospectives follow"),
        ("Actions", "Every retrospective produces at most three action items, each with a single named owner and a due date.",
         "how many action items a retrospective may produce"),
        ("Follow-up", "Open actions are reviewed in the first ten minutes of the next retrospective by the scrum master.",
         "how open retrospective actions get reviewed"))),
    ("design-review", "Design review", (
        ("Submission", "Design documents go into the architecture review channel at least three working days before the review meeting.",
         "how far ahead design documents go in"),
        ("Attendance", "The review panel always includes one engineer from a different team than the author of the design.",
         "who must sit on a design review panel"),
        ("Decision record", "Each decision is written up as an ADR in the docs repository and linked from the design document.",
         "where design decisions are recorded"))),
    ("customer-call-notes", "Customer call notes", (
        ("Recording", "Account managers log call notes in Salesforce within the same working day as the call.",
         "where call notes are logged"),
        ("Tagging", "Each note carries one of the tags risk, expansion or product feedback so that the weekly review can filter it.",
         "how call notes get tagged for review"),
        ("Sharing", "Notes marked confidential are visible only to the account owner and the regional director.",
         "who can read confidential call notes"))),
    ("support-shift-swaps", "Support shift swaps", (
        ("Requests", "Agents propose shift swaps to each other directly and post the agreed swap in the support rota channel.",
         "how support agents swap shifts"),
        ("Approval", "The team lead approves a swap only if both shifts stay covered by at least two agents with the Enterprise certification.",
         "who approves a support shift swap"),
        ("Records", "Approved swaps are entered in the scheduling tool by the lead on the same day.",
         "where approved shift swaps are recorded"))),
    ("ticket-macros", "Ticket macros", (
        ("Ownership", "Each macro has a named owner recorded in the macro library, and the owner is listed in the header comment.",
         "who owns a ticket macro"),
        ("Language", "Macros are written in the second person and never promise a date that engineering has not confirmed.",
         "how macro wording should address the customer"),
        ("Retirement", "A macro that has not been used in ninety days is hidden from the library by the support operations team.",
         "when an unused macro is hidden"))),
    ("conference-attendance", "Conference attendance", (
        ("Requests", "Conference requests are raised in the Lumen tracker by the employee at least a month before the event.",
         "how a conference request gets raised"),
        ("Sharing", "Each attendee gives a fifteen-minute readout in the engineering demo slot within two weeks of returning.",
         "how attendees share what they learned"),
        ("Budget", "Conference tickets are paid from the team learning line, and the travel is booked by the travel desk.",
         "which budget pays for conference tickets"))),
    ("shared-drive-folders", "Shared drive folders", (
        ("Naming", "Shared drive folders are named with the owning team first and the year last, separated by a single dash.",
         "how shared drive folders are named"),
        ("Sharing", "Folders are shared with groups rather than individuals, so that membership changes need no edits to the folder.",
         "how to share a folder with a team"),
        ("Cleanup", "Empty folders older than a year are moved into the Archive drive by the team admin during the quarterly tidy.",
         "what happens to old empty folders"))),
    ("one-on-ones", "One-on-one meetings", (
        ("Cadence", "Managers hold a one-on-one with each direct report every other week, and never cancel two in a row.",
         "how often managers meet direct reports one-on-one"),
        ("Notes", "The shared notes document for each one-on-one lives in Google Docs and is kept by the report.",
         "who keeps the one-on-one notes"),
        ("Agenda", "The report adds the agenda at least a day ahead, and the manager adds any career topics below it.",
         "who writes the agenda for a one-on-one"))),
    ("focus-time", "Focus time", (
        ("Blocks", "Engineers can mark two-hour focus blocks in the calendar on Tuesdays and Thursdays, which others must not book over.",
         "when engineers protect focus time"),
        ("Signals", "A focus block is signalled to the team by setting the status to heads-down in the chat app.",
         "how a team member shows they are heads-down"),
        ("Exceptions", "Only the on-call engineer or the incident commander may interrupt a focus block, and they do so by phone.",
         "who may interrupt focus time"))),
    ("security-training", "Security awareness training", (
        ("Schedule", "Every employee completes the security awareness course in the learning platform within their first two weeks.",
         "when new staff must finish security training"),
        ("Phishing drills", "Phishing drills are sent by the security team once a month, and anyone who clicks is enrolled in a refresher.",
         "what happens after a phishing drill click"),
        ("Reporting", "Suspicious emails are reported with the Report Phish button in Outlook, which forwards them to the security mailbox.",
         "how a suspicious email gets reported"))),
    ("lab-access", "Hardware lab access", (
        ("Booking", "Hardware lab bench time is booked through the lab calendar at least two working days ahead.",
         "how lab bench time is booked"),
        ("Safety", "Anyone using the lab must wear the safety glasses kept in the drawer beside the door.",
         "what to wear in the hardware lab"),
        ("Clearing", "Benches are cleared and wiped by the user before leaving, and the lab steward checks them at the end of each day.",
         "who checks the benches at the end of the day"))),
    ("team-channels", "Team channels", (
        ("Creation", "New team channels are requested through the Slack admin form and need a named owner before they are created.",
         "how new team channels get requested"),
        ("Archiving", "Channels with no messages for six months are archived by the workspace admin after a warning in the channel.",
         "when inactive channels get archived"),
        ("Naming", "Channel names start with the team prefix and use lowercase words separated by hyphens only.",
         "how channels should be named"))),
    ("mailing-list-etiquette", "Mailing list etiquette", (
        ("Distribution lists", "Company-wide distribution lists are reserved for the people team and the executive assistants.",
         "who may post to company-wide lists"),
        ("Subject lines", "Subject lines start with the team name in square brackets so that filters can route the message correctly.",
         "how mailing list subject lines are formatted"),
        ("Replies", "Reply-all is discouraged on lists with more than fifty members, and replies go to the sender unless asked otherwise.",
         "when reply-all is appropriate on a large list"))),
    ("demo-day", "Demo day", (
        ("Schedule", "Demo day runs on the last Thursday of each month, with each team given a fixed twelve-minute slot.",
         "when the monthly demo day takes place"),
        ("Recording", "Demos are recorded to the shared video library by the producer, and the recording is linked from the team page.",
         "where demo recordings get stored"),
        ("Feedback", "Feedback on demos is captured in the demo tracker by the product manager within two days.",
         "who collects feedback after a demo"))),
    ("accessibility-review", "Accessibility review", (
        ("Scope", "Every new screen in the web client passes a keyboard-only walkthrough before it reaches design review.",
         "what checks a new screen must pass first"),
        ("Tools", "Accessibility reviewers use the Axe browser extension and record each finding on the accessibility board.",
         "which tool accessibility reviewers use"),
        ("Sign-off", "A screen is signed off by the accessibility champion for the product area, not by the team that built it.",
         "who signs off accessibility on a screen"))),
    ("localization-requests", "Localization requests", (
        ("Intake", "Translation requests go to the localization queue in Phrase with the source string and its screenshot attached.",
         "how translation requests are submitted"),
        ("Vendors", "Only the two approved translation agencies may receive product strings, and procurement holds their contracts.",
         "which agencies may translate product strings"),
        ("Glossary", "Product terms are checked against the locale glossary, which the terminology lead updates every quarter.",
         "how product terms are kept consistent across languages"))),
    ("data-request-queue", "Internal data requests", (
        ("Intake", "Requests for internal data go through the analytics intake form, and the requester states the business question in one sentence.",
         "how internal data requests are submitted"),
        ("Turnaround", "The analytics team aims to answer routine requests within three working days and labels anything slower.",
         "how fast routine data requests are answered"),
        ("Access", "Raw customer tables are opened only to analysts who have finished the privacy course and sit on the data access list.",
         "who may open raw customer tables"))),
    ("brand-assets", "Brand assets", (
        ("Source files", "Logo and brand source files live in the design library on the shared drive, and nobody exports them from a screenshot.",
         "where the logo source files are kept"),
        ("Templates", "Slide templates are updated by the brand team only, and every deck starts from the official template file.",
         "where to start a new presentation deck"),
        ("Fonts", "The approved typefaces are installed on every company laptop by the IT desk and are not licensed for personal use.",
         "which typefaces are approved for company work"))),
)

TRANSCRIPTS = {
    "engineering": (
        ("Nightly build runners", "The nightly build moves to the ARM runner pool starting next sprint.", "moving the nightly build onto ARM runners"),
        ("Billing schema freeze", "Schema changes to the billing database freeze for the two weeks before quarter close.", "freezing billing database schema changes before quarter close"),
        ("Logging vendor", "The team keeps the current logging vendor and renegotiates the ingestion rate instead of switching.", "renegotiating ingestion rates instead of switching logging vendors"),
        ("Feature flag cleanup", "Feature flags older than six months are removed in the monthly cleanup, with no exception for experiments.", "removing feature flags older than six months"),
        ("Dependency pinning", "Application dependencies are pinned to exact versions, and the lockfile is reviewed in every release.", "pinning application dependencies to exact versions"),
        ("Weekend pager rota", "Two platform engineers join the weekend pager rotation to share the load with the core team.", "adding platform engineers to the weekend pager rotation"),
        ("Canary hold", "Every production deploy holds at the canary stage for at least one hour before full rollout.", "holding production deploys at canary before full rollout"),
        ("Mobile release train", "The mobile app ships on a fortnightly train, and hotfixes go out off-cycle only for crashes.", "shipping the mobile app on a fortnightly release train"),
        ("Load test environment", "The load test environment is torn down after each run instead of being kept warm.", "tearing down the load test environment after each run"),
        ("Code ownership file", "CODEOWNERS entries are generated from the team directory each night rather than edited by hand.", "generating code ownership entries from the team directory"),
        ("Secret scanning gate", "Secret scanning becomes a blocking check on every pull request to the main repository.", "making secret scanning a blocking pull request check"),
        ("Mobile repository split", "The payments service stays in the monorepo, and only the mobile client moves to its own repository.", "splitting the mobile client out of the monorepo"),
        ("Flaky test quarantine", "Intermittently failing tests are quarantined for no longer than a week before they are fixed or deleted.", "quarantining flaky tests"),
        ("Base image adoption", "Services adopt the hardened base image from the platform team by the end of next quarter.", "switching services to the hardened base image"),
        ("Incident channels", "Each incident gets a dedicated channel named after the incident, which is archived on closure.", "creating a channel for each incident"),
        ("Staging data refresh", "Staging receives a masked copy of production data on the first Monday of each month.", "refreshing staging with masked production data"),
        ("Public API versions", "Breaking changes to the public API ship only under a new version path, never in place.", "shipping breaking API changes under a new version path"),
        ("Shared cache eviction", "The shared cache evicts least recently used entries first instead of the oldest writes.", "changing the shared cache eviction policy"),
        ("Pairing policy", "Pairing becomes optional for the team, but auth changes still need a walkthrough for a reviewer.", "making pair programming optional"),
        ("Production debug logs", "Debug logging stays off in production by default and can be enabled per service for a single hour.", "enabling debug logs in production"),
        ("Client retry budget", "Client retries draw on a shared budget per service so that an outage does not multiply traffic.", "capping client retries with a shared budget"),
        ("Orders failover drills", "Failover drills for the orders database move from twice a year to every quarter.", "how often orders database failover drills run"),
        ("Terraform state backend", "Terraform state moves to a single remote backend with locking enabled for every workspace.", "moving Terraform state to one remote backend"),
        ("Edge rate limiting", "Rate limiting moves to the edge gateway so that internal services stop enforcing their own limits.", "moving rate limiting to the edge gateway"),
        ("Accessibility checks", "Automated accessibility checks run on every merge to the web client, and failures block the release.", "running accessibility checks on every merge"),
        ("Android crash sampling", "Crash reports from the Android client are sampled at a fixed rate to control storage costs.", "sampling Android crash reports"),
        ("Preview environments", "Pull requests get a preview environment only when they touch the frontend, to save compute.", "creating preview environments for pull requests"),
        ("Legacy SDK retirement", "The old client SDK is deprecated with a long notice period, after which requests on it are refused.", "deprecating the old client SDK"),
        ("Backup restore tests", "Restore tests for backups run every Friday in a throwaway account, and results go to the reliability channel.", "restore testing for backups"),
        ("Freeze exceptions", "Exceptions to the code freeze need a written note from the product owner and the release manager.", "who may approve code freeze exceptions"),
        ("Search reindexing window", "Reindexing of the search cluster is scheduled outside business hours and announced one day in advance.", "timing of search cluster reindexing"),
        ("Third-party SDK upgrades", "Upgrades to third-party SDKs happen in dedicated branches that never merge alongside feature work.", "upgrading third-party SDKs"),
        ("Metric naming", "New metrics must follow the team naming convention and are rejected at review if they break it.", "naming conventions for new metrics"),
        ("Mobile offline mode", "Offline mode for the mobile client waits until the sync service supports conflict resolution.", "shipping offline mode in the mobile client"),
        ("Deploy rollback control", "The deploy dashboard gets a one-click rollback that only the release manager and the on-call lead can use.", "who can trigger a one-click rollback"),
        ("Edge certificate renewal", "Certificates on the edge gateway are renewed by automation, with manual renewal kept only as a fallback.", "renewing edge gateway certificates"),
        ("Analytics batch cadence", "The analytics pipeline moves from nightly batch to hourly micro-batches for the product dashboards.", "moving the analytics pipeline to hourly batches"),
        ("Idle sandboxes", "Developer sandboxes idle for more than two days are suspended automatically each evening.", "suspending idle developer sandboxes"),
        ("Dead code cleanup", "Dead code is removed in a dedicated cleanup branch each release, never mixed into feature changes.", "removing dead code during releases"),
        ("Hiring exercise", "The engineering hiring process replaces the take-home project with a live paired exercise.", "replacing the take-home project in hiring"),
    ),
    "finance": (
        ("Month-end close target", "The month-end close targets business day five, and late journal entries need a written reason.", "the month-end close target day"),
        ("Accrual reversals", "Accruals reverse automatically on the first day of the following month for all open purchase orders.", "automatic reversal of accruals"),
        ("Corporate card reconciliation", "Corporate card spend is reconciled weekly by cardholders rather than monthly by the finance team.", "weekly corporate card spend reconciliation"),
        ("Revenue forecast cadence", "The revenue forecast is refreshed monthly with a rolling twelve-month horizon.", "how often the revenue forecast is refreshed"),
        ("Regional bank accounts", "Two of the five regional bank accounts close, and their balances move into the primary account.", "closing regional bank accounts"),
        ("Capitalisation threshold", "Equipment purchases below one thousand dollars are expensed rather than capitalised from this year.", "capitalising equipment purchases"),
        ("Prepaid software", "Prepaid software licences are amortised straight-line over their term, with no early write-off allowed.", "amortising prepaid software licences"),
        ("New vendor references", "New vendors need two bank references before their first payment is released by accounts payable.", "bank references for new vendors"),
        ("Meal receipts", "Receipts for meals must list the attendees, and claims without attendee names are returned unpaid.", "attendee names on meal receipts"),
        ("External audit sampling", "The external auditor samples invoices above a value threshold rather than at random this year.", "how external auditors sample invoices"),
        ("Multi-year contract review", "Multi-year contracts are reviewed by the controller each quarter before any revenue is recognised.", "reviewing multi-year contract revenue"),
        ("Sales tax filings", "Quarterly sales tax filings are prepared by the external firm two weeks before each deadline.", "who prepares quarterly sales tax filings"),
        ("Petty cash float", "The petty cash float is reduced, and any cash payment above a small threshold is refused.", "the petty cash float"),
        ("Budget variance explanations", "Department heads explain any variance above ten percent of budget in the monthly pack.", "explaining budget variances"),
        ("Intercompany loan terms", "Intercompany loans are documented with a written agreement and an interest rate set quarterly.", "documenting intercompany loans"),
        ("Bonus accrual timing", "Bonus accruals are booked in the quarter the performance year ends, not when payment is approved.", "when bonus accruals get booked"),
        ("Travel above policy cap", "Travel above the policy cap needs written approval from the department head and the finance partner.", "approving travel above the policy cap"),
        ("Hardware asset tags", "Every laptop and monitor receives an asset tag at receipt, and the tag is entered before the invoice is paid.", "tagging newly received hardware"),
        ("Purchase card limit resets", "Purchase card limits reset on the first of each month and cannot be raised mid-cycle.", "resetting purchase card limits"),
        ("Overdue customer holds", "Customers more than sixty days overdue are placed on credit hold by the collections team.", "placing overdue customers on credit hold"),
        ("Euro exposure hedging", "The company hedges half of its forecast euro exposure for the next four quarters.", "hedging euro exposure"),
        ("Office furniture depreciation", "Office furniture is depreciated over seven years on a straight-line basis from this year.", "depreciating office furniture"),
        ("Large invoice routing", "Invoices above the approval limit are routed to the budget holder before they reach accounts payable.", "routing large invoices for approval"),
        ("Bank reconciliation tool", "Bank reconciliations move to the new matching tool next quarter, with the spreadsheet kept as a backup.", "switching bank reconciliations to a matching tool"),
        ("Charitable donations", "Charitable donations above the threshold require a board vote recorded in the minutes.", "approving charitable donations"),
        ("Late fee waivers", "Late payment fees may be waived only by the controller and never by an account manager.", "who can waive late payment fees"),
        ("Stock compensation expense", "Stock-based compensation expense is recognised over the vesting period using the graded method.", "recognising stock-based compensation"),
        ("Cloud commitment shortfalls", "Unused cloud commitments are reviewed at each quarter end, and any shortfall is accrued at once.", "accruing unused cloud commitments"),
        ("Supplier payment runs", "Supplier payment runs happen on Tuesdays and Thursdays only, with urgent runs approved by the controller.", "the supplier payment run schedule"),
        ("Competing quotes", "Three competing quotes are required for any purchase above the tender threshold, unless a sole source is documented.", "quotes needed for large purchases"),
        ("Customer deposit refunds", "Customer deposits are refunded within fifteen business days after the contract is cancelled.", "refunding customer deposits after cancellation"),
        ("Board pack timing", "The board pack is circulated five working days before the meeting and marked confidential.", "when the board pack goes out"),
        ("Gym membership claims", "Gym memberships are removed from the approved expense categories and are no longer reimbursed.", "whether gym memberships are reimbursable"),
        ("Hardware counts", "Hardware inventory counts happen twice a year, with a spot check on the first Monday of each quarter.", "hardware inventory counting schedule"),
        ("Bad debt write-offs", "Bad debt write-offs above the controller's limit need sign-off from the chief financial officer.", "approving bad debt write-offs"),
        ("Office lease accounting", "Office leases are recorded on the balance sheet under the new standard using a single discount rate.", "recording office leases on the balance sheet"),
        ("Grant spending reports", "Grant spending reports are filed with the funder within a month of the end of each reporting period.", "filing grant spending reports"),
        ("Bank deposit limits", "Money market deposits are capped per counterparty so that no single bank holds more than a third of cash.", "per-bank deposit limits"),
        ("Unused subscriptions", "Unused software subscriptions are cancelled at renewal, after a review by the owning team lead.", "cancelling unused software subscriptions"),
        ("Year-end purchase cutoff", "The year-end cutoff for purchase invoices is the last working day of December, with no late additions.", "year-end cutoff for purchase invoices"),
    ),
    "hr": (
        ("Final panel makeup", "Every final interview panel includes one member from outside the hiring team.", "who sits on final interview panels"),
        ("Referral bonus timing", "The referral bonus is paid after the new hire completes probation, not at signature.", "when referral bonuses are paid"),
        ("Mid-year review timing", "Mid-year performance reviews move from July to the first week of September this cycle.", "timing of mid-year performance reviews"),
        ("Parental pay top-up", "The company tops up statutory parental pay to full salary for the first twelve weeks.", "top-up of parental pay"),
        ("Flexible working answers", "Flexible working requests are answered within two weeks, and any refusal is explained in writing.", "answering flexible working requests"),
        ("Learning allowance", "Each employee receives an annual learning allowance, which can be carried over only once.", "the annual learning allowance"),
        ("Senior engineer probation", "The probation period is extended for engineers hired at the senior level only.", "probation period for senior engineers"),
        ("Exit interview owner", "Exit interviews are conducted by the people partner, never by the departing employee's manager.", "who runs exit interviews"),
        ("Wellbeing days", "Staff receive one wellbeing day per quarter, which must be booked in the people system in advance.", "wellbeing days for staff"),
        ("Coaching programme entry", "The internal coaching programme is opened to people managers first, starting with the operations group.", "who may join the internal coaching programme"),
        ("Relocation housing", "Relocation support covers temporary housing for up to six weeks, after which the employee makes their own arrangements.", "temporary housing during relocation"),
        ("Bereavement documentation", "Bereavement leave for a close family member is granted without asking for a doctor's note.", "bereavement leave documentation"),
        ("Support holiday rota", "The holiday rota for the support desk is set by the team lead and published before the year starts.", "setting the support holiday rota"),
        ("Remote hire equipment", "New remote hires receive their equipment by courier and join a welcome call on their first Monday.", "how remote hires receive equipment"),
        ("Job title changes", "Job titles are standardised across the company, and any change needs approval from the people committee.", "approving job title changes"),
        ("Benefits enrolment close", "Benefits enrolment closes at the end of the first month, and late joiners wait for the next window.", "when benefits enrolment closes"),
        ("Sabbatical eligibility", "Sabbaticals are open to staff after five years of service, and each request must name a handover owner.", "who qualifies for a sabbatical"),
        ("Harassment training refresh", "Harassment prevention training is refreshed every year and tracked in the learning system by the people team.", "how often harassment training is refreshed"),
        ("Mentoring pair matching", "Mentoring pairs are matched by the people team and reviewed after the first quarter together.", "how mentoring pairs are assigned"),
        ("Home office claims", "Home office equipment is reimbursed up to a monthly cap, claimed through the people portal.", "reimbursing home office equipment"),
        ("Overtime logging", "Exempt staff do not record overtime, but non-exempt staff log every extra hour in the timesheet.", "recording overtime for non-exempt staff"),
        ("Reference checks", "Reference checks are carried out for every offer above the associate level by a member of the people team.", "who carries out reference checks"),
        ("Team lunch funding", "Team lunches are funded per head at a fixed amount, and receipts go to the people office.", "the per-head team lunch funding"),
        ("Office dress code", "The dress code is relaxed on office days, but client meetings still follow business attire.", "dress code for office days"),
        ("Grievance acknowledgement", "Formal grievances are acknowledged within three working days and resolved within a target of six weeks.", "grievance acknowledgement timeline"),
        ("Internal transfer approval", "Internal transfers need the current manager's agreement before the candidate can interview.", "internal transfer approval"),
        ("Retirement match rate", "The company retirement plan match rises from four percent to five percent for employees with two years of service.", "retirement plan match rate"),
        ("Volunteer day allowance", "Volunteer days are granted for each registered charity activity, up to two days per calendar year.", "volunteer day allowance"),
        ("New manager course", "New people managers complete the managers course within their first ninety days in the role.", "when new managers finish their course"),
        ("Hybrid attendance", "Teams are asked to be in the office together on Wednesdays, but individual attendance is not tracked.", "hybrid office attendance expectations"),
        ("Peer recognition awards", "Peer recognition awards are nominated through the feedback tool and voted on by a rotating panel.", "how peer recognition awards are chosen"),
        ("Contractor conversions", "Fixed-term contractors convert to permanent roles only after a formal competitive process.", "converting fixed-term contractors"),
        ("Resignation notice", "Staff who resign work a notice period of one month unless their contract states a different length.", "resignation notice length"),
        ("Pension opt-out", "Automatic pension enrolment happens after the first pay date, and staff can opt out within thirty days.", "opting out of pension enrolment"),
        ("Supplier gift declarations", "Gifts from suppliers worth more than a modest value must be declared to the people team.", "declaring supplier gifts"),
        ("Interview feedback deadline", "Interview feedback is submitted within twenty-four hours, before candidates are discussed as a panel.", "deadline for interview feedback"),
        ("Support shift swap rules", "Shift swaps on the support rota are agreed between colleagues and logged in the scheduling tool by the lead.", "agreeing support shift swaps"),
        ("Nursing room booking", "The quiet room on the third floor is booked through the facilities calendar for nursing parents.", "booking the quiet room for nursing parents"),
        ("Probation review points", "Probation reviews are held at the one-month and four-month marks, with the manager writing the summary.", "when probation reviews are held"),
        ("Diversity survey reporting", "Diversity survey responses are reported only in aggregate groups of at least ten people.", "reporting diversity survey results"),
    ),
}

RESTRICTED = (
    ("finance", "platform-engineering-budget", "Platform engineering budget",
     "The platform engineering team has an approved annual budget of 2,460,000 USD, and 1,130,000 USD of it is reserved for committed cloud spend.",
     "approved annual budget for platform engineering"),
    ("finance", "data-science-gpu-cap", "Data science GPU spending cap",
     "The data science team may spend up to 318,000 USD on GPU capacity before a second signature is required from finance.",
     "GPU spending cap for the data science team"),
    ("finance", "customer-success-hiring-funds", "Customer success hiring funds",
     "Customer success is funded for fourteen additional account managers, at a total cost of 790,000 USD for the current year.",
     "funding for extra customer success account managers"),
    ("finance", "security-ops-contingency", "Security operations contingency pool",
     "Security operations holds a contingency pool of 95,000 USD for emergency vendor engagements that are not in the plan.",
     "security operations contingency pool"),
    ("finance", "field-sales-travel-cap", "Field sales travel cap",
     "Field sales travel is capped at 240,000 USD for the second half of the year, including the regional summit.",
     "field sales travel cap for the second half"),
    ("finance", "marketing-agency-retainer", "Marketing agency retainer",
     "Marketing operations has 610,000 USD earmarked for the agency retainer, and the renewal terms are still under negotiation.",
     "amount earmarked for the marketing agency retainer"),
    ("finance", "legal-research-budget", "Legal research budget",
     "The legal team's outside research budget is 77,500 USD and cannot be moved to litigation costs without the general counsel.",
     "outside research budget of the legal team"),
    ("finance", "eastgate-lease-renegotiation", "Eastgate office lease renegotiation",
     "Facilities has set aside 1,200,000 USD for the Eastgate office lease renegotiation, which stays confidential until signed.",
     "amount set aside for the Eastgate lease renegotiation"),
    ("finance", "developer-relations-sponsorship", "Developer relations sponsorship",
     "Developer relations has 185,000 USD for the partner conference sponsorship in the spring, approved on a conditional basis.",
     "sponsorship money for the spring partner conference"),
    ("finance", "revenue-ops-forecasting-tool", "Revenue operations forecasting tool",
     "Revenue operations received 42,000 USD for a new forecasting tool, with a formal review scheduled for March.",
     "budget for the new revenue forecasting tool"),
    ("finance", "northgate-termination-fee", "Northgate Data Centers termination fee",
     "The Northgate Data Centers colocation contract carries an early termination fee of 880,000 USD and a 36-month minimum commitment.",
     "early termination fee in the Northgate Data Centers contract"),
    ("finance", "harlow-fuel-surcharge", "Harlow Freight fuel surcharge cap",
     "The Harlow Freight hardware logistics contract caps annual fuel surcharges at 6.5 percent of the base freight charge.",
     "fuel surcharge cap in the Harlow Freight contract"),
    ("finance", "brightwater-overage-rate", "Brightwater Telecom overage rate",
     "Brightwater Telecom charges 0.9 cents per outbound minute above the committed pool under the contract signed last year.",
     "per-minute overage rate in the Brightwater Telecom contract"),
    ("finance", "quillstone-notice-period", "Quillstone Analytics licence notice",
     "The Quillstone Analytics licence renews automatically unless written notice is given 120 days before the anniversary date.",
     "notice needed to end the Quillstone Analytics licence"),
    ("finance", "orchard-service-credit", "Orchard Payroll service credit",
     "Orchard Payroll Services guarantees a 99.95 percent payroll success rate and pays a credit of 4,000 USD when it misses that target.",
     "service credit owed by Orchard Payroll Services"),
    ("finance", "pemberton-volume-rebate", "Pemberton Office Supplies rebate",
     "Pemberton Office Supplies offers a volume rebate of 3 percent once annual spend passes 150,000 USD, paid as a credit note.",
     "volume rebate agreed with Pemberton Office Supplies"),
    ("finance", "vireo-onboarding-charge", "Vireo Security Scanning onboarding charge",
     "The Vireo Security Scanning agreement includes unlimited rescans but excludes a one-off onboarding charge of 24,000 USD.",
     "onboarding charge in the Vireo Security Scanning agreement"),
    ("finance", "asterline-credit-expiry", "Asterline Cloud Credits expiry",
     "Asterline Cloud Credits expire 18 months after purchase and cannot be transferred from one account to another.",
     "expiry period for Asterline cloud credits"),
    ("finance", "marlowe-minimum-order", "Marlowe Print and Ship minimum order",
     "Marlowe Print and Ship requires a minimum monthly order of 2,500 USD under the current statement of work.",
     "minimum monthly order for Marlowe Print and Ship"),
    ("finance", "calder-rate-clause", "Calder Legal Research rate clause",
     "Calder Legal Research may raise hourly rates by up to 5 percent each January without prior notice to the client.",
     "hourly rate increase clause in the Calder Legal Research contract"),
    ("finance", "solenne-exclusivity", "Solenne Catering exclusivity clause",
     "The Solenne Catering agreement bars any other caterer from the Ridgeway site for two years from signature.",
     "exclusivity clause in the Solenne Catering agreement"),
    ("finance", "corvid-early-return", "Corvid Fleet Leasing early return charge",
     "Corvid Fleet Leasing charges 1,450 USD per vehicle returned before the end of the lease term.",
     "early return charge for Corvid Fleet Leasing vehicles"),
    ("finance", "bluestone-retention", "Bluestone Insurance cyber policy retention",
     "The Bluestone Insurance cyber policy has a 58,000 USD annual premium and a 250,000 USD self-insured retention.",
     "retention amount in the Bluestone Insurance cyber policy"),
    ("finance", "gridline-sla-credit", "Gridline Network Services SLA credit",
     "Gridline Network Services grants an SLA credit only after three consecutive monthly breaches of the uptime commitment.",
     "SLA credit condition in the Gridline Network Services contract"),
    ("hr", "senior-analyst-band-l5", "Salary band: senior data analyst, level L5",
     "The salary band for a senior data analyst at level L5 runs from 118,000 to 142,000 USD, with the midpoint held back for calibration.",
     "salary band for a senior data analyst at level L5"),
    ("hr", "staff-sre-band-l6", "Salary band: staff site reliability engineer, level L6",
     "A staff site reliability engineer at level L6 is banded between 150,000 and 181,000 USD base salary.",
     "pay range for a staff site reliability engineer at level L6"),
    ("hr", "associate-ae-band-l2", "Salary band: associate account executive, level L2",
     "An associate account executive at level L2 has a base band of 62,000 to 79,000 USD, plus commission set separately.",
     "base pay band for an associate account executive at level L2"),
    ("hr", "principal-designer-band-l7", "Salary band: principal product designer, level L7",
     "The principal product designer band at level L7 is 176,000 to 210,000 USD, reviewed once a year by the compensation committee.",
     "salary band for a principal product designer at level L7"),
    ("hr", "support-specialist-band-l1", "Salary band: support specialist, level L1",
     "Support specialists at level L1 are paid within a band of 44,000 to 52,000 USD, with shift premiums added on top.",
     "pay band for a support specialist at level L1"),
    ("hr", "eng-manager-band-m2", "Salary band: engineering manager, level M2",
     "An engineering manager at level M2 sits in a band of 170,000 to 198,000 USD, and the top of the band needs director sign-off.",
     "salary band for an engineering manager at level M2"),
    ("hr", "finance-partner-band-l4", "Salary band: finance business partner, level L4",
     "A finance business partner at level L4 is banded at 96,000 to 118,000 USD, with the upper half reserved for promotions.",
     "salary band for a finance business partner at level L4"),
    ("hr", "security-analyst-band-l3", "Salary band: security analyst II, level L3",
     "The security analyst II band at level L3 is 88,000 to 105,000 USD, and the bonus target is set at twelve percent.",
     "salary band for a security analyst II at level L3"),
    ("hr", "head-cs-band-e1", "Salary band: head of customer success, level E1",
     "The head of customer success at level E1 is banded between 205,000 and 240,000 USD, before equity.",
     "salary band for the head of customer success at level E1"),
    ("hr", "technical-writer-band-l3", "Salary band: technical writer, level L3",
     "Technical writers at level L3 have a band of 79,000 to 92,000 USD, and the band was last moved in the spring review.",
     "salary band for a technical writer at level L3"),
    ("hr", "recruiter-band-l3", "Salary band: recruiter, level L3",
     "A recruiter at level L3 is paid within 73,000 to 87,000 USD, and agency-sourced hires do not change that band.",
     "salary band for a recruiter at level L3"),
    ("hr", "legal-counsel-band-l5", "Salary band: legal counsel, level L5",
     "Legal counsel at level L5 sits within a band of 165,000 to 188,000 USD, and the band excludes any retention award.",
     "salary band for legal counsel at level L5"),
    ("hr", "leave-germany", "Leave rules: Germany",
     "Employees in Germany receive 30 paid leave days a year, and up to 10 unused days may carry forward into the next year.",
     "paid leave allowance for employees in Germany"),
    ("hr", "leave-portugal", "Leave rules: Portugal",
     "Portugal staff receive 25 leave days plus two optional days that must be booked by the last day of December.",
     "leave entitlement for staff based in Portugal"),
    ("hr", "leave-japan", "Leave rules: Japan",
     "Japan staff accrue leave on the statutory schedule and may take up to 5 unpaid sick days each year without a certificate.",
     "unpaid sick day allowance for staff in Japan"),
    ("hr", "leave-brazil", "Leave rules: Brazil",
     "Brazilian employees receive 30 days of vacation, and the 13th salary is paid in the second half of November.",
     "vacation and 13th salary rule for Brazil"),
    ("hr", "leave-ontario", "Leave rules: Ontario",
     "Ontario staff receive 3 personal days a year, and the entitlement resets on the anniversary of each hire date.",
     "personal day allowance for Ontario staff"),
    ("hr", "leave-australia", "Leave rules: Australia",
     "Australian employees accrue four weeks of annual leave, loaded to five weeks for shift workers on the roster.",
     "annual leave loading for Australian shift workers"),
    ("hr", "leave-india", "Leave rules: India",
     "India staff receive 18 earned leave days, and up to 9 of them can be encashed once per year at the basic rate.",
     "earned leave encashment for India staff"),
    ("hr", "leave-mexico", "Leave rules: Mexico",
     "Mexico staff receive 15 vacation days in year one, and the allowance rises by two days on each anniversary.",
     "vacation schedule for Mexico staff"),
    ("hr", "leave-poland", "Leave rules: Poland",
     "Poland staff receive a four-day bridge allowance around national holidays, and any unused bridge days expire on March 31.",
     "bridge days allowance for Poland staff"),
    ("hr", "leave-netherlands", "Leave rules: Netherlands",
     "Dutch employees may buy up to 10 extra leave days a year through the salary sacrifice scheme administered by payroll.",
     "leave purchase scheme for Dutch employees"),
    ("hr", "leave-singapore", "Leave rules: Singapore",
     "Singapore employees receive 14 days of annual leave, and a medical certificate is required for any sick leave after the first day.",
     "medical certificate rule for sick leave in Singapore"),
    ("hr", "leave-ireland", "Leave rules: Ireland",
     "Irish staff receive 20 days of annual leave plus public holidays, with a parental leave top-up for eight weeks.",
     "parental leave top-up for Irish staff"),
)

# Legacy documents for the superseding topics are generated from the topic table.
# These 60 retired announcements cover subjects that no curated topic, ticket, handbook page,
# transcript or restricted document uses.
LEGACY = (
    ("friday-bagel-fund", "Retired: Friday bagel fund",
     "The Friday bagel fund paid for breakfast on the first Friday of each month for anyone in the office. Contributions were collected in a tin by the office manager and spent at the corner bakery. The fund ended when the office moved to a shared kitchen with its own snack budget."),
    ("old-intranet-home", "Retired: old intranet home page",
     "The old intranet home page listed every department and its shared folder in one long list. Pages were edited by hand in a tool that no longer has a licence. Links to the old home page should point to the wiki index, and the page itself has been switched off."),
    ("vendor-fax-line", "Retired: fax line for vendors",
     "Vendors who could not send invoices electronically were asked to use the office fax line. Faxes were printed at the front desk and scanned into the shared folder each afternoon. The line was closed once every active supplier had moved to the portal."),
    ("parking-allocation", "Retired: parking allocation scheme",
     "Car park spaces were allocated by seniority under a scheme run from the facilities office. Spaces were reshuffled every spring and holders showed a permit at the barrier. The scheme was replaced by first come booking in the facilities app."),
    ("office-plant-rota", "Retired: office plant care rota",
     "Each floor had a rota for watering the potted plants, and the names were chalked on a board beside the lifts. Volunteers kept a note of any plant that was struggling. The rota was dropped when the plants were replaced with artificial ones."),
    ("kitchen-snack-rota", "Retired: kitchen snack rota",
     "A rota decided who bought the biscuits for the kitchen each week. The person on duty posted a receipt on the board beside the fridge. The rota stopped once the office manager began ordering a standing snack box. Anyone who wanted the rota back could ask the office manager."),
    ("badge-photo-day", "Retired: badge photo day",
     "Staff once queued in the atrium on a set day to have new photos taken for their access badges. The photographer worked from a fixed backdrop in the hall. The photo day was discontinued when the badge office began accepting uploaded portraits."),
    ("book-club", "Retired: book club",
     "The book club met in the library on the last Wednesday of each month to discuss a shared novel. Members took turns choosing titles and bringing a short summary. The group disbanded when most members moved to other sites. Anyone interested in restarting it should speak to the library team."),
    ("company-newsletter", "Retired: monthly company newsletter format",
     "The monthly newsletter used a long printed format with a column of news from each department. Editors collected items by email and laid the pages out by hand. The newsletter became a short weekly digest in the wiki, and the print run was stopped."),
    ("org-chart-binder", "Retired: printed org chart binder",
     "A binder with a printed org chart sat on the reception desk and was updated by the admin team every quarter. Visitors were allowed to browse it while they waited. The binder was retired once the chart moved into the staff directory."),
    ("cafeteria-vendor", "Retired: cafeteria vendor switch notice",
     "The ground floor cafeteria changed operator at the start of the year, and the old menu boards were removed. Staff were asked to return their old meal cards to the front desk. The notice was kept for reference and no longer applies to the building."),
    ("suggestion-box", "Retired: suggestion box",
     "A cardboard suggestion box stood beside the lifts on the second floor, and its contents were read at the monthly staff meeting. Ideas were answered on a noticeboard by the operations lead. The box was removed after the staff survey took over the same role."),
    ("meeting-room-names", "Retired: meeting room naming scheme",
     "Meeting rooms were once named after rivers, and the names were printed on small plates outside each door. The facilities team replaced the plates when the floors were reorganised. Old room names still appear in some calendar entries, but they should not be used."),
    ("desk-booking-app", "Retired: desk booking app",
     "A small booking app let staff reserve a hot desk for a day or a week. The app stopped working when the vendor withdrew support for the platform it ran on. Staff now sit wherever a free desk is available on their floor."),
    ("bicycle-storage", "Retired: bicycle storage rules",
     "Bicycles could be stored in the basement racks under a set of rules posted near the entrance. Owners had to tag each bicycle with a name plate and clear abandoned bikes each month. The racks were reorganised and the old rules were withdrawn."),
    ("all-hands-format", "Retired: all-hands meeting format",
     "All-hands meetings once ran as a single long session with a slide deck and a question queue. Speakers were given fifteen minutes each and questions were taken from paper cards. The format was replaced by a shorter session with pre-recorded updates."),
    ("holiday-party-vote", "Retired: holiday party theme vote",
     "Each year the staff voted on a theme for the holiday party in an online poll run by the events group. The winning theme set the decorations, music and dress suggestions. The vote was dropped when the party moved to a rotating venue format."),
    ("email-signature", "Retired: email signature template",
     "All staff once used a shared email signature template with a photo, a title and a set of legal lines at the bottom. Local edits were not allowed and the template was pushed by the mail administrator. The template was withdrawn when the mail system began managing signatures itself."),
    ("mail-room-collection", "Retired: mail room collection procedure",
     "Internal post was collected from the mail room by a runner who walked each floor twice a day. Recipients signed a book when they took delivery of parcels. The runner role was removed when the building switched to a locker system for deliveries."),
    ("shared-printer-queue", "Retired: shared printer queue",
     "The shared printer on each floor used a numbered queue, and jobs were released by swiping a staff card. Long jobs often blocked smaller ones for minutes at a time. The queue was retired when the floor printers were replaced by secure release stations."),
    ("dress-up-fridays", "Retired: dress-up Fridays",
     "Staff could dress up on Fridays in exchange for a small donation to a rotating charity. The charity was chosen by a vote held in the first week of each month. The arrangement ended when the company adopted a single relaxed dress policy for the whole week."),
    ("courier-list", "Retired: approved courier list",
     "Staff sending urgent documents were asked to use one of three approved couriers listed on the intranet. Bookings were made through a shared inbox. The list was withdrawn after the courier contracts expired and were not renewed. Teams that still need a courier now book one through the general enquiries inbox."),
    ("standing-desk-pool", "Retired: standing desk loan pool",
     "A small pool of standing desk converters could be borrowed for a month at a time from the facilities store. Borrowers signed a sheet and returned the converter when the loan ended. The pool was closed once every floor had its own adjustable desks."),
    ("tea-subscription", "Retired: tea subscription",
     "The kitchen once received a monthly box of loose leaf tea from a local supplier, and the box was shared on each floor. Staff could add a request to the notes pad by the kettle. The subscription lapsed and the supplier was not replaced."),
    ("lanyard-colour-code", "Retired: lanyard colour code",
     "Visitor and contractor lanyards were colour coded so that reception staff could tell them apart at a glance. The colour chart was printed on the back of the reception door. The code was dropped when visitors began wearing printed badges instead."),
    ("quarterly-picnic", "Retired: quarterly picnic",
     "The quarterly picnic was held in the park by the river and included games, a raffle and a barbecue. Families were welcome and the events group booked a marquee each time. The picnic was cancelled after the park reserved the space for a public festival."),
    ("pigeonhole-mail", "Retired: pigeonhole mail system",
     "Each staff member had a pigeonhole at the mail room for paper notices and small items. Notices were sorted every morning by a clerk who checked the names against the staff list. The pigeonholes were removed when the building was refitted."),
    ("recycling-contractor", "Retired: recycling contractor",
     "The old recycling contractor collected paper and plastic from every floor on a fixed weekly route. Bins were labelled by colour and the contractor issued a monthly report to facilities. The contract ended and the bins were replaced with a new collection scheme."),
    ("lobby-display-screens", "Retired: lobby display screens",
     "Screens in the lobby once showed a rotating set of news items, weather and a live feed of the atrium. Content was uploaded from a shared folder by the communications team. The screens were switched off when the building signage was redesigned."),
    ("team-nicknames", "Retired: team nicknames list",
     "A list of informal team nicknames was kept on the intranet so that people could find each other quickly in chat. Entries were added by each team lead and reviewed once a year. The list was retired when the chat directory began showing team names directly."),
    ("open-office-playlist", "Retired: open office playlist policy",
     "The open office once played a shared music playlist over the speakers during the afternoon. Anyone could suggest a track through a form at the front desk. The speakers were switched off after complaints about the volume during calls. Playlists from the old form were deleted at the same time."),
    ("lost-property-log", "Retired: lost property log",
     "Lost items found in the building were recorded in a paper log kept at the security desk. Owners could claim items by signing the log and showing identification. The paper log was replaced by a simple shared spreadsheet and then closed."),
    ("travel-mug-giveaway", "Retired: travel mug giveaway",
     "New joiners once received a steel travel mug with the company logo on the first day. The mugs were stored in a cupboard on the ground floor and handed out by reception. The giveaway stopped when the stock ran out and was not restocked."),
    ("swimming-pool-discount", "Retired: swimming pool discount scheme",
     "Staff could claim a discount at the local swimming pool by showing a card from the facilities desk. The scheme was negotiated by the wellbeing group for a single year. The discount lapsed when the pool changed owners. Staff who had used the card were asked to hand it back."),
    ("wall-calendar", "Retired: wall calendar",
     "Each floor had a large paper wall calendar where teams marked off days out of the office. The calendar was updated by a designated person each Monday. The calendars were taken down when the team calendars moved to a shared app."),
    ("standing-meeting-timer", "Retired: standing meeting timer",
     "Meeting rooms once had a sand timer on the table to keep standing meetings short. Teams were asked to finish when the sand ran out. The timers were removed when rooms began showing a countdown on the wall screen. Teams that still wanted the habit kept a stopwatch on their own desk."),
    ("elevator-booking", "Retired: elevator booking",
     "The freight elevator could be reserved for moving furniture through the facilities calendar. Bookings were made by email and confirmed by the building team. Reservations were discontinued when the lift was replaced with a model that does not need booking."),
    ("pinboard-awards", "Retired: pinboard awards",
     "Small awards were once pinned to a board in the canteen for staff who had helped a colleague that week. Nominations were written on slips of paper and placed in a tin. The board was removed when the canteen was refurbished."),
    ("community-garden-plots", "Retired: community garden plots",
     "Staff were offered small plots in the community garden at the back of the site, and each plot had a name tag on the gate. Gardeners shared tools kept in a shed by the path. The plots were cleared when the land was sold to a neighbouring school."),
    ("intern-welcome-tour", "Retired: intern welcome tour",
     "Interns once took a guided walk around the building on their first morning, led by a volunteer from each floor. The tour covered the kitchens, the library and the emergency exits. The tour was replaced by an online introduction video."),
    ("desk-plant-swap", "Retired: desk plant swap",
     "Staff swapped spare desk plants twice a year in a bright corner of the atrium. Plant cuttings were labelled with the name of the donor. The swap ended when the atrium was closed for building works. Plants that had not been claimed went to the reception area."),
    ("breakroom-fridge-labels", "Retired: breakroom fridge labels",
     "Food in the breakroom fridge had to carry a label with the owner's name and the date it was stored. Unlabelled items were cleared out every Friday. The labelling rule was dropped when the fridges were replaced with lockable units."),
    ("airport-shuttle", "Retired: airport shuttle service",
     "A shared minibus once took staff from the central station to the airport for early departures. Seats were booked through the front desk on a first come basis. The service was withdrawn when the bus contract came to an end."),
    ("staff-choir", "Retired: staff choir practice",
     "The staff choir rehearsed in the auditorium every Thursday evening and sang at two charity events each year. Membership was open to anyone who could attend regularly. Practice stopped when the auditorium was given over to training rooms. Members who wished to continue met in a local church hall instead."),
    ("bulletin-board-rules", "Retired: bulletin board rules",
     "The corridor bulletin boards carried a set of rules about what could be pinned and for how long. Notices had to show a name and a removal date in the corner. The rules were withdrawn when notices moved to the intranet."),
    ("staff-portrait-sessions", "Retired: staff portrait sessions",
     "Professional portraits were taken for the staff directory during scheduled sessions in a small studio on the third floor. Staff booked a slot with the communications team. The sessions ended once the directory began using avatars chosen by each person."),
    ("stationery-key-log", "Retired: stationery cupboard key log",
     "The stationery cupboard was locked, and the key was signed out from a log held by the office manager. Requests for pens and paper were written on a slip. The log was dropped when the cupboard was opened to all staff."),
    ("toner-ordering", "Retired: printer toner ordering process",
     "Toner cartridges were ordered through a paper form that the floor champion posted to the facilities office. Orders were grouped and sent once a week to the supplier. The process was replaced when the printers were leased with automatic supply."),
    ("cycle-to-work-scheme", "Retired: cycle-to-work scheme",
     "The cycle-to-work scheme let staff buy a bicycle and accessories through deductions from pay over several months. Quotes were gathered from two local shops. The scheme closed to new applicants when the provider stopped offering it. Staff already enrolled kept their existing payment schedule until it ended."),
    ("stand-up-roulette", "Retired: daily stand-up roulette",
     "For a time the daily stand-up was run by a different person each morning, chosen by a random wheel in the team room. The leader kept the meeting to ten minutes. The roulette was dropped when teams chose their own fixed facilitators."),
    ("whiteboard-colour-code", "Retired: whiteboard marker colour code",
     "Whiteboards in the studio followed a colour code, with blue for ideas, red for blockers and green for decisions. Each marker was labelled on the tray. The code was abandoned when the studio whiteboards were replaced with digital screens. Old markers were given to the art room."),
    ("game-night", "Retired: game night schedule",
     "Game nights were held on the second Tuesday of the month in the breakout room, with board games and card games on offer. Teams signed up on a sheet by the door. The schedule was cancelled after the breakout room was booked for training."),
    ("ceiling-projector-booking", "Retired: ceiling projector booking",
     "Rooms with ceiling projectors could be reserved through the old room booking sheet at the front of each floor. Bulbs were replaced by the facilities team on request. The projectors were removed when the rooms were fitted with wall screens."),
    ("franking-machine", "Retired: franking machine procedure",
     "Outgoing post was franked on a machine in the mail room, and the postage was charged back to each cost centre. Staff brought their envelopes to the desk before noon. The machine was retired once outgoing post moved to a contracted provider."),
    ("team-poster-competition", "Retired: team poster competition",
     "Each year teams were invited to produce a poster about their work, and the best posters were displayed in the foyer. Judges were drawn from other departments. The competition was stopped when the foyer was converted to a reception lounge."),
    ("paper-courier-forms", "Retired: paper courier forms",
     "Urgent packages were sent using a paper form that listed the sender, recipient and reason for the urgency. The form was signed by a manager and passed to the mail room. The forms were withdrawn when the courier booking tool was introduced."),
    ("canteen-menu-rotation", "Retired: canteen menu rotation",
     "The canteen menu rotated through a set of four weekly plans, with a vegetarian option on each day. Feedback was collected on cards placed beside the till. The rotation was replaced by a new menu owned by the catering supplier."),
    ("charity-bake-sale", "Retired: charity bake sale",
     "A charity bake sale was held in the atrium each spring, with cakes donated by staff and sold at a fixed price. Proceeds went to a children's hospital charity chosen by the events group. The sale ended when the atrium was closed for refurbishment."),
    ("reception-greeting-script", "Retired: reception greeting script",
     "Receptionists followed a short greeting script when answering the front door phone and welcoming guests. The script was printed on a card by the desk and revised each spring. The card was removed when the reception desk was moved to the new lobby."),
    ("stairwell-signage", "Retired: stairwell signage project",
     "A project once replaced the painted floor numbers in the stairwells with large enamel signs. The contractor used the old colour scheme for consistency. The project was paused when the stairwells were closed for a full structural survey. Staff on the upper floors were asked to use the lifts during the work."),
)

NOTE_DISCUSSION = (
    "The group compared the options against the figures from last quarter before settling.",
    "Most of the time went on the risks, and the owner agreed to write them up for the next review.",
    "Two alternatives were set aside early because they did not fit the tooling the team already runs.",
)
NAMES = ("Priya", "Tomas", "Ngozi", "Ben", "Lucia", "Hiro", "Mariam", "Sven", "Aisha", "Dmitri", "Chloe", "Rafael")
TEAM_LABEL = {"engineering": "engineering", "finance": "finance", "hr": "HR"}

UNANSWERABLE = (
    "Does Kestrel Orbit support offline replication for field crews?",
    "What does the Kestrel Mesh add-on cost for industrial sensors?",
    "Who runs the sister company of Kestrel Cloud in Brazil?",
    "Which export formats does Kestrel Quill offer to legal teams?",
    "Has Kestrel Cloud ever opened a data center on the Moon?",
    "What costume did the mascot wear at the Kestrel Cloud 2029 conference?",
    "Is there a company-wide four-day workweek pilot at Kestrel Cloud?",
    "How many shares did Kestrel Cloud repurchase during its first public offering?",
    "Which airline holds the executive travel contract with Kestrel Cloud?",
    "Does Kestrel Cloud sponsor a motorsport team in a European championship?",
    "When does the Kestrel Ember hardware refresh begin?",
    "Where is the Kestrel Cloud research laboratory in Nairobi located?",
    "What does the pet insurance benefit at Kestrel Cloud cover?",
    "Which university holds the quantum computing chair that Kestrel Cloud funds?",
    "Does Kestrel Cloud accept cryptocurrency for supplier invoices?",
    "Can third-party drone telemetry plugins be listed on the Kestrel Cloud Marketplace?",
    "What is the reimbursement cap for scuba diving courses under the wellness plan?",
    "Who won the internal chess tournament at the Lisbon office?",
    "What is the paternity bonus for employees based in Auckland?",
    "Which satellite ground station network does Kestrel Cloud rent time from?",
    "What uptime guarantee does the Kestrel Orbit service carry?",
    "What does a mainframe connector license cost at Kestrel Cloud?",
    "Which EU country hosts the sovereign cloud region for defence customers?",
    "How long are voice recordings kept on the Kestrel Quill phone line?",
    "Is there a lifetime license for the desktop Kestrel Mesh client?",
    "Where was the 2027 annual shareholder meeting of Kestrel Cloud held?",
    "What is the after-hours hotline for employees in the Lagos office?",
    "Which hospital provides on-site vaccinations at the Denver campus?",
    "Who designed the logo that Kestrel Cloud used before its newest identity?",
    "Does Kestrel Cloud run a bug bounty for its wearable companion app?",
    "What is the executive car lease scheme in Singapore?",
    "Which Kestrel product connects to the Orbit telemetry bus?",
    "Is there a loyalty card for frequent business travellers at Kestrel Cloud?",
    "How many robots operate in the Kestrel Cloud warehouse in Rotterdam?",
    "What brand color is mandated for Kestrel Ember dashboards?",
    "Which bank provides the corporate credit facility in Japan?",
    "Does Kestrel Cloud accept Bitcoin payment for enterprise invoices?",
    "What share of flight emissions does Kestrel Cloud offset each year?",
    "Who chairs the ethics board that meets in Oslo?",
    "What hours does the on-site childcare centre keep at the Tokyo office?",
    "Which podcast featured the Kestrel Cloud CTO last spring?",
    "Is Kestrel Cloud a member of the Helix Alliance trade group?",
    "How long is the waiting period for the Kestrel Mesh firmware beta program?",
    "Does Kestrel Cloud offer massage vouchers at the Porto office?",
    "What coffee break interval is mandated in the Lisbon warehouse?",
    "Who approved the acquisition of the Berlin robotics startup?",
    "What penalty clause appears in the Kestrel Orbit reseller agreement?",
    "Which chip architecture powers the Kestrel Ember edge device?",
    "Does Kestrel Cloud run a robotics competition for schoolchildren each summer?",
    "What is the company policy on drone deliveries to customers' homes?",
)

EXACT_Q = ("What is the {ask} at Kestrel Cloud?", "What does Kestrel Cloud set as the {ask}?")
SUPERSEDED_Q = ("What is the current {ask} at Kestrel Cloud?", "Which {ask} is in force today at Kestrel Cloud?")
CURATED_Q = ("What is the process for {ask}?", "How does Kestrel Cloud handle {ask}?")
TICKET_Q = ("How did support resolve the {product} case about {topic}?", "What was the remedy for {topic} on {product}?",
            "How was the problem with {topic} on {product} fixed?")
HANDBOOK_Q = ("What does the handbook say about {ask}?", "Which handbook page covers {ask}?")
TRANSCRIPT_Q = ("What did the {team} team decide about {ask}?", "Which decision came out of the {team} meeting on {ask}?")
RESTRICTED_Q = ("What is the {ask}?", "Can you tell me the {ask}?")
# Near misses ask about a covered topic for a fact the topic's document does not state.
NEAR_MISS_Q = ("In which year did Kestrel Cloud introduce the {ask}?", "How many times has Kestrel Cloud changed the {ask}?")
ID_CLASSES = ("exact-", "superseded-", "curated-", "expired-", "ticket-", "handbook-", "transcript-", "strict-",
              "denied-ticket-", "denied-restricted-", "unanswerable-", "nearmiss-")


def _front(fields: list[str]) -> str:
    return "---\n" + "\n".join(fields) + "\n---\n"


def _old_value(rng: random.Random, new: int) -> int:
    # The legacy page must not contain the new value as a substring of its own digits.
    while True:
        old = rng.randint(11, 997)
        if old != new and str(new) not in str(old):
            return old


def _curated(docs: dict[str, str], values: dict[str, int]) -> None:
    for t in TOPICS:
        value = values[t.slug]
        review = EXPIRED_REVIEW_BY if t.slug in EXPIRED else REVIEW_BY
        fields = [f"type: {t.area}", f"title: {json.dumps(t.title)}", f"owner: {json.dumps(OWNERS[t.group])}",
                  f"review_by: {review}", f"acl: [{t.group}]"]
        if t.slug in SUPERSEDING:
            fields.append(f"supersedes: [archive/legacy/{t.slug}-old]")
        fields += ["facts:", f"  {t.key}: {{value: {value}, unit: {json.dumps(t.unit)}}}"]
        body = " ".join([t.fact.format(v=value), t.procedure, t.filler_a, t.filler_b])
        docs[f"knowledge/{t.area}/{t.slug}.md"] = _front(fields) + f"# {t.title}\n\n{body}\n"


def _legacy(docs: dict[str, str], rng: random.Random, values: dict[str, int]) -> int:
    count = 0
    for t in TOPICS:
        if t.slug not in SUPERSEDING:
            continue
        old = _old_value(rng, values[t.slug])
        body = " ".join([t.fact.format(v=old), t.procedure, "This page was retired after the replacement took effect."])
        docs[f"archive/legacy/{t.slug}-old.md"] = f"# {t.title} (retired)\n\n{body}\n"
        count += 1
    for slug, title, body in LEGACY:
        docs[f"archive/legacy/{slug}.md"] = f"# {title}\n\n{body}\n"
        count += 1
    return count


def _ticket_text(rng: random.Random, issue: tuple) -> str:
    _, product, symptom, cause, resolution, _ = issue
    customer, plan, day = rng.choice(CUSTOMERS), rng.choice(PLANS), rng.choice(DAYS)
    layout = rng.randrange(3)
    if layout == 0:
        opener = rng.choice(OPENERS_A).format(customer=customer, plan=plan, day=day, symptom=symptom)
        sentences = [opener, f"Our engineer found {cause} during the first call.",
                     f"We {resolution}, and the customer confirmed the problem had stopped.",
                     "The ticket was closed the same day with a note in its history."]
    elif layout == 1:
        sentences = [f"Customer: {customer} ({plan} plan). Reported on {day}: {symptom}.",
                     f"Root cause: {cause}.", f"Resolution: we {resolution}."]
    else:
        opener = rng.choice(OPENERS_C).format(customer=customer, plan=plan, day=day, symptom=symptom)
        sentences = [opener, f"Investigation showed {cause}.",
                     f"After the team {resolution}, the symptom stopped and the customer agreed to close the case."]
    title = f"{product}: {symptom}"
    return _front([f"title: {json.dumps(title)}", "acl: [support]"]) + f"# {title}\n\n" + " ".join(sentences) + "\n"


def _tickets(docs: dict[str, str], rng: random.Random) -> dict[str, list[str]]:
    numbers = rng.sample(range(10000, 100000), len(ISSUES) * 30)
    ids: dict[str, list[str]] = {}
    for k, issue in enumerate(ISSUES):
        ids[issue[0]] = []
        for i in range(30):
            doc_id = f"archive/tickets/T-{numbers[k * 30 + i]:05d}"
            docs[doc_id + ".md"] = _ticket_text(rng, issue)
            ids[issue[0]].append(doc_id)
    return ids


def _handbook(docs: dict[str, str]) -> list[tuple[str, int, str]]:
    practices = []
    for slug, title, sections in HANDBOOK:
        parts = [_front([f"title: {json.dumps(title)}", "acl: [public]"]), f"# {title}\n"]
        for n, (heading, text, ask) in enumerate(sections, start=1):
            parts.append(f"## {heading}\n\n{text}\n")
            practices.append((f"archive/handbook/{slug}", n, ask))
        docs[f"archive/handbook/{slug}.md"] = "\n".join(parts)
    return practices


def _transcripts(docs: dict[str, str], rng: random.Random) -> list[tuple[str, str, str]]:
    entries = []
    for team, rows in TRANSCRIPTS.items():
        for n, (topic, decision, ask) in enumerate(rows, start=1):
            when = (REFERENCE - timedelta(days=rng.randint(5, 240))).isoformat()
            a, b, c = rng.sample(NAMES, 3)
            discussion = rng.choice(NOTE_DISCUSSION)
            title = f"Meeting notes: {topic}"
            doc_id = f"archive/transcripts/{team}-{n:03d}"
            docs[doc_id + ".md"] = (
                _front([f"title: {json.dumps(title)}", f"acl: [{team}]"])
                + f"# {title}\n\nDate: {when}. Attendees: {a}, {b} and {c}.\n\n{discussion}\n\nDecision: {decision}\n")
            entries.append((doc_id, team, ask))
    return entries


def _restricted(docs: dict[str, str]) -> list[tuple[str, str, str]]:
    entries = []
    for group, slug, title, sentence, ask in RESTRICTED:
        docs[f"archive/restricted/{slug}.md"] = (
            _front([f"title: {json.dumps(title)}", f"acl: [{group}]"]) + f"# {title}\n\n{sentence}\n")
        entries.append((f"archive/restricted/{slug}", group, ask))
    return entries


def _case(case_id: str, question: str, answerable: bool, tags: tuple[str, ...], *, groups: tuple[str, ...] = (),
          min_authority: str | None = None, gold: tuple[str, ...] = (), must: tuple[str, ...] = (),
          forbidden: tuple[str, ...] = ()) -> dict:
    row: dict = {"id": case_id, "question": question, "answerable": answerable}
    if groups:
        row["groups"] = list(groups)
    if min_authority:
        row["min_authority"] = min_authority
    row["tags"] = list(tags)
    if gold:
        row["gold_docs"] = list(gold)
    if must:
        row["must_contain"] = list(must)
    if forbidden:
        row["forbidden_docs"] = list(forbidden)
    return row


def _assign_splits(rows: list[dict]) -> None:
    seen: Counter[str] = Counter()
    for row in rows:
        cls = max((c for c in ID_CLASSES if row["id"].startswith(c)), key=len)
        row["split"] = "fit" if seen[cls] % 2 == 0 else "holdout"
        seen[cls] += 1


def _golden(rng: random.Random, values: dict[str, int], tickets: dict[str, list[str]],
            handbook: list[tuple[str, int, str]], transcripts: list[tuple[str, str, str]],
            restricted: list[tuple[str, str, str]], seed: int) -> list[dict]:
    rows: list[dict] = []
    plain = [t for t in TOPICS if t.slug not in SUPERSEDING and t.slug not in EXPIRED]
    for t in plain:
        rows.append(_case(f"exact-{t.slug}", rng.choice(EXACT_Q).format(ask=t.ask), True, ("exact", "curated"),
                          groups=() if t.group == "public" else (t.group,),
                          gold=(f"kb/{t.area}/{t.slug}",), must=(str(values[t.slug]),)))
    for t in TOPICS:
        if t.slug in SUPERSEDING:
            rows.append(_case(f"superseded-{t.slug}", rng.choice(SUPERSEDED_Q).format(ask=t.ask), True,
                              ("exact", "curated", "superseded"), gold=(f"kb/{t.area}/{t.slug}",),
                              must=(str(values[t.slug]),), forbidden=(f"archive/legacy/{t.slug}-old",)))
    picked = set(rng.sample(plain, 30))
    for t in plain:
        if t in picked:
            rows.append(_case(f"curated-{t.slug}", rng.choice(CURATED_Q).format(ask=t.proc_ask), True, ("curated",),
                              groups=() if t.group == "public" else (t.group,),
                              gold=(f"kb/{t.area}/{t.slug}",)))
    for t in TOPICS:
        if t.slug in EXPIRED:
            rows.append(_case(f"expired-{t.slug}", rng.choice(EXACT_Q).format(ask=t.ask), False, ("expired",),
                              min_authority="curated", forbidden=(f"kb/{t.area}/{t.slug}",)))
    for issue in ISSUES:
        rows.append(_case(f"ticket-{issue[0]}", rng.choice(TICKET_Q).format(topic=issue[5], product=issue[1]), True,
                          ("archive",), groups=("support",), gold=tuple(tickets[issue[0]])))
    picked_h = set(rng.sample(range(len(handbook)), 10))
    for index in sorted(picked_h):
        page, n, ask = handbook[index]
        rows.append(_case(f"handbook-{page.rsplit('/', 1)[1]}-{n}", rng.choice(HANDBOOK_Q).format(ask=ask), True,
                          ("archive",), gold=(page,)))
    picked_t = set(rng.sample(range(len(transcripts)), 20))
    for index in sorted(picked_t):
        doc_id, team, ask = transcripts[index]
        rows.append(_case(f"transcript-{doc_id.rsplit('/', 1)[1]}",
                          rng.choice(TRANSCRIPT_Q).format(team=TEAM_LABEL[team], ask=ask), True, ("archive",),
                          groups=(team,), gold=(doc_id,)))
    for issue in rng.sample(ISSUES, 20):
        rows.append(_case(f"strict-{issue[0]}", rng.choice(TICKET_Q).format(topic=issue[5], product=issue[1]), False,
                          ("min_authority",), groups=("support",), min_authority="curated",
                          forbidden=tuple(tickets[issue[0]])))
    for issue in rng.sample(ISSUES, 30):
        rows.append(_case(f"denied-ticket-{issue[0]}", rng.choice(TICKET_Q).format(topic=issue[5], product=issue[1]),
                          False, ("acl_denied",), forbidden=tuple(tickets[issue[0]])))
    for doc_id, _group, ask in rng.sample(restricted, 20):
        rows.append(_case(f"denied-restricted-{doc_id.rsplit('/', 1)[1]}", rng.choice(RESTRICTED_Q).format(ask=ask),
                          False, ("acl_denied",), groups=("support",), forbidden=(doc_id,)))
    for n, question in enumerate(UNANSWERABLE, start=1):
        rows.append(_case(f"unanswerable-{n:02d}", question, False, ("unanswerable",)))
    chosen = set(random.Random(seed + 1).sample(plain, 30))
    for n, t in enumerate(x for x in plain if x in chosen):
        rows.append(_case(f"nearmiss-{t.slug}", NEAR_MISS_Q[n % 2].format(ask=t.ask), False, ("near_miss",),
                          groups=() if t.group == "public" else (t.group,)))
    _assign_splits(rows)
    return rows


def generate(out: Path, golden: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    docs: dict[str, str] = {}
    values = dict(zip((t.slug for t in TOPICS), rng.sample(range(11, 998), len(TOPICS))))
    _curated(docs, values)
    legacy = _legacy(docs, rng, values)
    tickets = _tickets(docs, rng)
    handbook = _handbook(docs)
    transcripts = _transcripts(docs, rng)
    restricted = _restricted(docs)
    rows = _golden(rng, values, tickets, handbook, transcripts, restricted, seed)

    for root in ("knowledge", "archive"):
        shutil.rmtree(out / root, ignore_errors=True)
    for rel in sorted(docs):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(docs[rel], encoding="utf-8", newline="\n")
    golden.parent.mkdir(parents=True, exist_ok=True)
    golden.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8", newline="\n")
    return {
        "curated": len(TOPICS),
        "legacy": legacy,
        "tickets": sum(len(v) for v in tickets.values()),
        "handbook": len(HANDBOOK),
        "transcripts": len(transcripts),
        "restricted": len(restricted),
        "golden": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "corpus")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    counts = generate(args.out, args.golden, args.seed)
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
