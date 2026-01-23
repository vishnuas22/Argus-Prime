"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Copy, Link, Calendar, Check, Loader2 } from "lucide-react";
import { reportApi } from "@/lib/api";
import { useToast } from "@/components/ui/toast";

interface ShareLinkProps {
  analysisId: string;
  onShare?: () => void;
}

interface ShareLinkState {
  shareUrl: string;
  expiresAt: string;
  isLoading: boolean;
  isCopied: boolean;
}

export function ShareLink({ analysisId, onShare }: ShareLinkProps) {
  const [state, setState] = useState<ShareLinkState>({
    shareUrl: "",
    expiresAt: "",
    isLoading: false,
    isCopied: false,
  });
  const { addToast } = useToast();

  const generateLink = async () => {
    setState((prev) => ({ ...prev, isLoading: true }));
    try {
      const result = await reportApi.getShareableLink(analysisId);
      setState({
        shareUrl: result.shareUrl,
        expiresAt: result.expiresAt,
        isLoading: false,
        isCopied: false,
      });
      onShare?.();
    } catch {
      setState((prev) => ({ ...prev, isLoading: false }));
      addToast("error", "Error", "Failed to generate shareable link");
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(state.shareUrl);
      setState((prev) => ({ ...prev, isCopied: true }));
      addToast("success", "Copied", "Share link copied to clipboard");
      setTimeout(() => {
        setState((prev) => ({ ...prev, isCopied: false }));
      }, 2000);
    } catch {
      addToast("error", "Error", "Failed to copy link");
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Link className="h-5 w-5" />
          Share Results
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {!state.shareUrl ? (
          <Button
            onClick={generateLink}
            disabled={state.isLoading}
            className="w-full"
          >
            {state.isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Link className="mr-2 h-4 w-4" />
                Generate Shareable Link
              </>
            )}
          </Button>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Input
                value={state.shareUrl}
                readOnly
                className="flex-1"
                aria-label="Shareable link"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={copyToClipboard}
                className={state.isCopied ? "bg-green-50 text-green-600" : ""}
              >
                {state.isCopied ? (
                  <Check className="h-4 w-4" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Calendar className="h-4 w-4" />
              <span>Expires: {new Date(state.expiresAt).toLocaleString()}</span>
            </div>
            <Button
              variant="outline"
              onClick={generateLink}
              disabled={state.isLoading}
              className="w-full"
            >
              Generate New Link
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
