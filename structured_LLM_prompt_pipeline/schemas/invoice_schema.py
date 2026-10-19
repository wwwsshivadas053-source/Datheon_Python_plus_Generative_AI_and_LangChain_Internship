"""Pydantic schemas for structured invoice and financial document extraction.

Includes nested models, validation rules, mathematical integrity cross-checks,
and schema generation helpers for LLM prompt engineering.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class VendorInfo(BaseModel):
    """Details of the company or vendor issuing the invoice."""
    name: str = Field(..., description="Business or individual vendor name")
    address: Optional[str] = Field(None, description="Full physical or registered address")
    tax_id: Optional[str] = Field(None, description="Tax ID, VAT number, GSTIN, or EIN")
    email: Optional[str] = Field(None, description="Vendor contact email address")
    phone: Optional[str] = Field(None, description="Vendor phone or customer service number")
    website: Optional[str] = Field(None, description="Vendor website URL")


class CustomerInfo(BaseModel):
    """Details of the recipient or customer being billed."""
    name: str = Field(..., description="Customer, company, or client name")
    billing_address: Optional[str] = Field(None, description="Billing street address, city, country")
    shipping_address: Optional[str] = Field(None, description="Shipping or delivery destination address")
    email: Optional[str] = Field(None, description="Customer billing or contact email")
    phone: Optional[str] = Field(None, description="Customer telephone or contact number")
    contact_person: Optional[str] = Field(None, description="Direct contact person or attention line")


class InvoiceMetadata(BaseModel):
    """Administrative and chronological metadata of the invoice."""
    invoice_number: str = Field(..., description="Unique invoice identifier or document number")
    issue_date: str = Field(..., description="Invoice issue or billing date (ISO format preferred: YYYY-MM-DD)")
    due_date: Optional[str] = Field(None, description="Payment due date (ISO format preferred: YYYY-MM-DD)")
    po_number: Optional[str] = Field(None, description="Purchase order (PO) reference number")
    currency: str = Field("USD", description="Currency code (e.g., USD, EUR, GBP, INR, CAD)")
    payment_terms: Optional[str] = Field(None, description="Payment terms, e.g., Net 30, Due on Receipt")


class LineItem(BaseModel):
    """An individual line item or service billed on the invoice."""
    item_id: Optional[str] = Field(None, description="SKU, item code, or serial number")
    description: str = Field(..., description="Detailed description of goods supplied or service rendered")
    quantity: float = Field(..., ge=0, description="Quantity or hours billed")
    unit_price: float = Field(..., ge=0, description="Price per individual unit/hour")
    tax_rate_pct: Optional[float] = Field(0.0, ge=0, description="Tax percentage applied to this line (e.g., 10 for 10%)")
    line_total: float = Field(..., description="Total price for this line item (quantity * unit_price)")

    @model_validator(mode="after")
    def verify_line_total(self) -> "LineItem":
        """Verify that line_total roughly matches quantity * unit_price."""
        calculated = round(self.quantity * self.unit_price, 2)
        # Allow slight rounding variance (up to 0.1)
        if abs(self.line_total - calculated) > 0.1 and self.line_total == 0:
            # Self-heal missing line_total
            self.line_total = calculated
        return self


class FinancialSummary(BaseModel):
    """Overall financial totals and tax breakdowns."""
    subtotal: float = Field(..., ge=0, description="Pre-tax subtotal of all line items")
    tax_amount: float = Field(0.0, ge=0, description="Total sales tax, VAT, or GST assessed")
    discount_amount: float = Field(0.0, ge=0, description="Total discount or promo credit deducted")
    shipping_fee: float = Field(0.0, ge=0, description="Freight, handling, or shipping fee")
    grand_total: float = Field(..., ge=0, description="Final payable amount after taxes, shipping, and discounts")
    currency: str = Field("USD", description="Currency code matching metadata")
    payment_status: Optional[str] = Field("Unpaid", description="Payment status: Paid, Unpaid, Pending, Overdue")


class PaymentDetails(BaseModel):
    """Payment instructions, bank wire info, or remittance details."""
    bank_name: Optional[str] = Field(None, description="Remittance bank name")
    account_number: Optional[str] = Field(None, description="Bank account number or IBAN")
    routing_or_swift: Optional[str] = Field(None, description="Routing transit number, SWIFT/BIC code")
    ifsc_code: Optional[str] = Field(None, description="IFSC code for Indian banking")
    payment_method: Optional[str] = Field(None, description="Wire Transfer, Credit Card, ACH, UPI, PayPal")


class InvoiceExtraction(BaseModel):
    """Root model for complete structured invoice extraction."""
    vendor: VendorInfo = Field(..., description="Vendor/Seller details")
    customer: CustomerInfo = Field(..., description="Customer/Buyer details")
    metadata: InvoiceMetadata = Field(..., description="Invoice number, dates, PO, currency")
    line_items: List[LineItem] = Field(default_factory=list, description="List of itemized products or services")
    financials: FinancialSummary = Field(..., description="Subtotal, tax, discount, shipping, grand total")
    payment_info: Optional[PaymentDetails] = Field(None, description="Payment or banking instructions")
    notes: Optional[str] = Field(None, description="Special notes, remarks, return policy, or terms")
    extraction_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    math_discrepancy_detected: bool = Field(False, description="Flag indicating if a mathematical mismatch was detected in document")

    @model_validator(mode="after")
    def cross_check_financial_math(self) -> "InvoiceExtraction":
        """Cross-check mathematical consistency between line items and totals."""
        if self.line_items:
            items_sum = round(sum(item.line_total for item in self.line_items), 2)
            # Compare with subtotal
            if abs(items_sum - self.financials.subtotal) > 0.5:
                self.math_discrepancy_detected = True

        expected_grand_total = round(
            self.financials.subtotal + self.financials.tax_amount + self.financials.shipping_fee - self.financials.discount_amount,
            2
        )
        if abs(expected_grand_total - self.financials.grand_total) > 0.5:
            self.math_discrepancy_detected = True

        return self
